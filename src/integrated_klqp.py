
from enum import Enum
import numpy as np
import tensorflow as tf
from edward1_utils import get_ancestors, get_descendants


class GenerativeMode(Enum):

    UNCONDITIONED = 1  # i.e. sampling the learnt prior
    CONDITIONED = 2  # i.e. sampling the posterior, with variational samples substituted
    RECONSTRUCTION = 3  # i.e. mode of the posterior


def noncopying_integrated_reparam_klqp(generative_builder, variational_builder, name_to_data_map, discrete_name_to_states_map, sample_count=1, beta=1., alpha=5.e6, grad_clip_magnitude=None):

    # Every variable in the generative and variational should have a leading dimension that is 'IID', corresponding to
    # an index-into-batch or otherwise sampled independently -- when we make substitutions, this dimension may be
    # expanded to incorporate more samples. Thus, all RVs are indexed by iid-index, *

    # Generative RVs are created by lambdas, taking zero or one parameters. There should be zero parameters when
    # dim0 (i.e. the iid-dimension) has size fixed by ancestral variables; there should be one parameter when it's a 'root'
    # variable (i.e. doesn't have any ancestor-RVs) and its base dim0 should be multiplied by that parameter

    # Variational RVs are created similarly; the name given to the lambda should match that of the corresponding
    # generative variable. Sample/discrete-expanded observations are retrived with a second lambda

    # generative_builder is free to return any type or None; for example, it may choose to return an object containing
    # some of its random variables; the unconditioned and mode-reconstruction versions of this result are returned to
    # the caller

    # ** note that we do not allow (but do not check for!) non-leaf non-RV tensors that are iib-indexed and have an RV as a sibling to
    # ** be included as parents of  RVs in the graph, as these cannot easily be expanded to the correct dimensionality -- their iid-index
    # ** will always be of 'base' size, and will not broadcast correctly against 'upsampled' iid-indices of the sibling RV
    # ** this could be fixed by handling such things essentially the same as expansion-like-discrete

    # ** note that having RVs created with a non-default (i.e. not scalar) sample_shape will not work in general, as we call rv.sample()
    # ** directly without passing this in -- so the shape will not be what the caller expects

    assert len(discrete_name_to_states_map) < 2
    if len(discrete_name_to_states_map) == 1:
        assert False # ...as this is broken for now -- discrete states get 'blurred together'
        discrete_name, discrete_states = discrete_name_to_states_map.items()[0]  # discrete_states is a numpy array indexed by discrete-index, *
        assert discrete_name not in name_to_data_map
    else:
        discrete_name = None
        discrete_states = np.zeros([1])

    # Build the 'prior', i.e. the generative without variational substitutions, so we can evaluate the prior probability of the variational samples later
    name_to_unconditioned_generative_variable = {}
    generative_root_variable_names = set()
    def make_unconditioned_rv(name, builder):
        with tf.name_scope(name):
            assert name not in name_to_unconditioned_generative_variable
            is_root_variable = builder.__code__.co_argcount == 1  # ideally the below assert would *define* root-ness (indeed, it does, conceptually), but can't evaluate it before the variable is created!
            variable = builder(1) if is_root_variable else builder()
            assert is_root_variable == (len(get_ancestors(variable, name_to_unconditioned_generative_variable.values())) == 0)  # ** could be made more efficient by caching, so quickly know chunks of the graph do/don't have ancestor-RVs
            if is_root_variable:
                generative_root_variable_names.add(name)
            name_to_unconditioned_generative_variable[name] = variable
            return variable.value
    with tf.variable_scope('generative'), tf.name_scope('unconditioned'):
        unconditioned_generative = generative_builder(make_unconditioned_rv, GenerativeMode.UNCONDITIONED)

    def expand_like_discrete(substituted_value):
        # This will be applied to all variables that aren't indexed by discrete-state
        substituted_value = tf.reshape(substituted_value, [sample_count, -1] + list(map(int, substituted_value.get_shape()[1:])))  # indexed by sample-index, iid-index, *
        substituted_value = tf.tile(substituted_value, [1, discrete_states.shape[0]] + [1] * (substituted_value.get_shape().ndims - 2))  # indexed by sample-index, discrete-index * iid-index, *
        return tf.reshape(substituted_value, [-1] + list(map(int, substituted_value.get_shape()[2:])))  # indexed by sample-index * discrete-index * iid-index, *

    name_to_substituted_value = {}  # each value is indexed by sample-index * discrete-index * iid-index, *

    # Construct expanded copies of the observations (tiled over sample and discrete indices); these are made available
    # to the variational so it can reason over them, and are used as substitutions in the generative
    for name in name_to_data_map:
        assert name != discrete_name  # ** need to think about this case!
        # ** should also probably assert that the observed variable is not a variational-descendant of the discrete (or any other variable!)
        substituted_value = tf.tile(
            name_to_data_map[name],
            [sample_count] + [1] * (name_to_data_map[name].get_shape().ndims - 1)
        )  # indexed by sample-index * iid-index, *
        # ** is calling expand_like_discrete not strictly less efficient that just adding the discrete-state-count into the above tile?
        name_to_substituted_value[name] = expand_like_discrete(substituted_value)  # always expand, as an observed variable cannot be variational-descendant of the discrete

    def is_variable_discrete_indexed(variable):
        # Substituted values are always discrete-indexed, hence having one of them as an ancestor is a sufficient
        # condition for being discrete-indexed. In practice we check the reverse, as the substitution is not an RV
        # hence won't be returned as an ancestor. It is also a necessary condition, as there is no other route through
        # which discrete-indexing can be added
        return any(
            len(get_descendants(substituted_value, [variable])) > 0
            for substituted_value in name_to_substituted_value.values()
        )

    # Build the variational, substituting samples and expanding all variables to be indexed by sample and discrete indices
    name_to_conditioned_variational_variable = {}
    def make_variational_rv(name, builder):
        with tf.name_scope('q_' + name):
            assert name in name_to_unconditioned_generative_variable
            assert name not in name_to_data_map
            assert name not in name_to_conditioned_variational_variable
            is_root_variable = builder.__code__.co_argcount == 1  # ideally the below assert would *define* root-ness (indeed, it does, conceptually), but can't evaluate it before the variable is created!
            variable = builder(sample_count) if is_root_variable else builder()
            assert is_root_variable == (
                len(get_ancestors(variable, name_to_conditioned_variational_variable.values())) == 0  # it's a root variable if it doesn't have any variational RV as an ancestor...
                and
                all(
                    len(get_descendants(name_to_substituted_value[observation_name], [variable])) == 0  # ...and no observation has it as a descendant -- i.e. it doesn't have any observation as an ancestor either
                    for observation_name in name_to_data_map
                )
            )  # ** could be made more efficient by caching, so quickly know chunks of the graph do/don't have ancestor-RVs
            substituted_value = variable.value  # indexed by sample-index * [discrete-index *] iid-index, *
            if discrete_name is not None:  # if there's a discrete to be integrated, then *all* substituted values must be discrete-indexed
                if name == discrete_name:
                    assert map(int, substituted_value.get_shape()[1:]) == list(discrete_states.shape[1:])  # check the discrete values have the same shape as samples from the distribution
                    substituted_value = tf.tile(
                        discrete_states[np.newaxis, :, np.newaxis, ...],
                        [sample_count, 1, int(substituted_value.get_shape()[0]) / sample_count / (discrete_states.shape[0] if is_variable_discrete_indexed(variable) else 1)] + [1] * (len(discrete_states.shape) - 1)
                    )  # indexed by sample-index, discrete-index, iid-index, *
                    substituted_value = tf.reshape(substituted_value, [-1] + list(discrete_states.shape[1:]))  # indexed by sample-index * discrete-index * iid-index, *
                else:
                    if not is_variable_discrete_indexed(variable):
                        substituted_value = expand_like_discrete(substituted_value)
            name_to_conditioned_variational_variable[name] = variable  # this is used to evaluate the variational density of the variational sample; for both this and next, uses ancestral substitutions in case of non-MF variational
            name_to_substituted_value[name] = substituted_value  # this is substituted into the generative
            return substituted_value
    with tf.variable_scope('variational'), tf.name_scope('conditioned'):
        variational_builder(make_variational_rv, lambda observation_name: name_to_substituted_value[observation_name])
    if discrete_name is not None:
        assert discrete_name in name_to_conditioned_variational_variable
        assert discrete_name in name_to_substituted_value

    # Build the 'conditioned generative', with values substituted from the variational and observations
    name_to_conditioned_generative_variable = {}
    def make_conditioned_rv(name, builder):
        with tf.name_scope(name):
            is_root_variable = name in generative_root_variable_names  # i.e. whether this is an RV with no ancestor-RVs, meaning that it should be replicated according to sample_count (otherwise, replication of some ancestor should 'bubble down' to us)
            variable = builder(sample_count) if is_root_variable else builder()
            name_to_conditioned_generative_variable[name] = variable  # used to evaluate the generative density of the variational sample (and the observed data), with ancestral substitutions
            if name not in name_to_substituted_value:
                # Marginalise by sampling from the generative (with ancestral conditioning), as there's no corresponding variational or observation
                # ** could condition the warning on whether it actually has descendants!
                print('warning: {} has neither variational distribution nor observed value, hence will be marginalised by sampling'.format(name))
                substituted_value = variable.value
                if discrete_name is not None:
                    if not is_variable_discrete_indexed(variable):
                        substituted_value = expand_like_discrete(substituted_value)
                name_to_substituted_value[name] = substituted_value
            return name_to_substituted_value[name]
    with tf.variable_scope('generative', reuse=True), tf.name_scope('conditioned'):
        conditioned_generative = generative_builder(make_conditioned_rv, GenerativeMode.CONDITIONED)
    if discrete_name is not None:
        assert discrete_name in name_to_conditioned_generative_variable

    def get_mode_or_mean(variable):
        try:
            return variable.distribution.mode()
        except NotImplementedError:
            print('warning: using mean instead of mode for {} in reconstruction'.format(variable.distribution.name))
            return variable.distribution.mean()  # fall back to mean, e.g. for uniform random variables

    # Build a second copy of the variational, with the (variational) mode of each variable substituted, in order to do
    # a full 'ancestrally modal' reconstruction in the non-MF case
    name_to_variational_mode = {}
    def make_variational_reconstruction_rv(name, builder):
        with tf.name_scope('q_' + name):
            assert name in name_to_unconditioned_generative_variable
            is_root_variable = builder.__code__.co_argcount == 1  # ** cache from first variational model creation above?
            variable = builder(1) if is_root_variable else builder()
            name_to_variational_mode[name] = get_mode_or_mean(variable)
            return name_to_variational_mode[name]
    with tf.variable_scope('variational', reuse=True), tf.name_scope('modes'):
        variational_builder(make_variational_reconstruction_rv, lambda observation_name: name_to_data_map[observation_name])

    # This third copy of the generative is not used by inference, but is returned to the caller to use for reconstructions
    # It does not perform any sample/discrete expansion, but substitutes variational modes for ancestral latents
    def make_reconstruction_rv(name, builder):
        with tf.name_scope(name):
            if name in name_to_variational_mode:
                return name_to_variational_mode[name]
            else:
                # ** non-use of name_to_data_map here may not be desirable if the variable is not a leaf
                variable = builder(1) if name in generative_root_variable_names else builder()
                return get_mode_or_mean(variable)
    with tf.variable_scope('generative', reuse=True), tf.name_scope('reconstruction'):
        reconstruction_modes = generative_builder(make_reconstruction_rv, GenerativeMode.RECONSTRUCTION)

    with tf.name_scope('integrated_klqp'):

        def lifted_log_prob(variable, value, name):  # ** would be nice if we could rely on variable.name == name!
            # variable is a random variable, indexed by sample-index * [discrete-index *] iid-index, *
            # value is a tensor, indexed by sample-index * discrete-index * iid-index, *
            # This function evaluates variable.log_prob on slices of value taken over discrete-index, summing away non-iid dimensions
            discrete_state_count = discrete_states.shape[0]
            if discrete_name is None:
                log_prob = variable.distribution.log_prob(value)
                return tf.reduce_sum(log_prob, axis=list(range(1, log_prob.get_shape().ndims)))[np.newaxis, ...]
            elif is_variable_discrete_indexed(variable):
                log_prob = variable.distribution.log_prob(value)  # indexed by sample-index * discrete-index * iid-index, *
                log_prob = tf.reduce_sum(log_prob, axis=list(range(1, log_prob.get_shape().ndims)))  # indexed by sample-index * discrete-index * iid-index
                log_prob = tf.reshape(log_prob, [sample_count, discrete_state_count, -1])  # indexed by sample-index, discrete-index, iid-index
                return tf.reshape(tf.transpose(log_prob, [1, 0, 2]), [discrete_state_count, -1])  # indexed by discrete-index, sample-index * iid-index
            else:
                value = tf.reshape(value, [sample_count, discrete_state_count, -1] + list(map(int, value.get_shape()[1:])))  # indexed by sample-index, discrete-index, iid-index, *
                value = tf.transpose(value, [1, 0, 2] + list(range(3, value.get_shape().ndims)))  # indexed by discrete-index, sample-index, iid-index, *
                value = tf.reshape(value, [discrete_state_count, -1] + list(map(int, value.get_shape()[3:])))  # indexed by discrete-index, sample-index * iid-index, *
                log_prob = tf.stack([
                    variable.distribution.log_prob(value[state_index])
                    for state_index in range(discrete_state_count)
                ])  # indexed by discrete-index, sample-index * iid-index, *
                return tf.reduce_sum(log_prob, axis=range(2, log_prob.get_shape().ndims))  # indexed by discrete-index, sample-index * iid-index

        if discrete_name is not None:
            discrete_qz_probs = tf.exp(lifted_log_prob(
                name_to_conditioned_variational_variable[discrete_name],
                name_to_substituted_value[discrete_name],  # this is the discrete states tiled over sample-index and iid-index
                discrete_name
            ))  # indexed by discrete-index, sample-index * iid-index; this is the probability under the variational, of each discrete state
        def E_log_prob_wrt_discrete(variable, value, name):  # ** again, would be nice if could rely on variable.name == name!
            # log_prob is indexed by sample-index * [discrete-index *] iid-index, *
            # result is scalar, being a mean over samples, and minibatch-elements, an expectation over discrete-states, and a sum over remaining dimensions
            maybe_weighted_log_prob = lifted_log_prob(variable, value, name)  # indexed by discrete-index, sample-index * iid-index
            if discrete_name is not None:
                maybe_weighted_log_prob *= discrete_qz_probs
            return tf.reduce_mean(maybe_weighted_log_prob)  # that we do a mean over iid-index means we treat the minibatch-indexing as independent sampling, not a joint rv

        log_Px = sum(
            E_log_prob_wrt_discrete(name_to_conditioned_generative_variable[name], name_to_substituted_value[name], name)
            for name in name_to_data_map
        )
        log_Pz = sum(
            E_log_prob_wrt_discrete(name_to_conditioned_generative_variable[name], name_to_substituted_value[name], name)
            # for name in name_to_conditioned_generative_variable
            for name in name_to_conditioned_variational_variable  # variational not generative so we only include things with variational (not prior) substitutions
            if name != discrete_name  # ...as we use L1 divergence for this instead
            if name not in name_to_data_map  # ...as it's in P(x) instead
        )
        log_Qz = sum(
            E_log_prob_wrt_discrete(name_to_conditioned_variational_variable[name], name_to_substituted_value[name], name)
            for name in name_to_conditioned_variational_variable
            if name != discrete_name  # ...as we use L1 divergence for this instead
        )

        for name in name_to_conditioned_variational_variable:
            if name != discrete_name:
                if name not in name_to_data_map:
                    tf.summary.scalar('P(z_' + name + ')', E_log_prob_wrt_discrete(name_to_conditioned_generative_variable[name], name_to_substituted_value[name], name))
        for name in name_to_data_map:
            tf.summary.scalar('P(x_' + name + ')', E_log_prob_wrt_discrete(name_to_conditioned_generative_variable[name], name_to_substituted_value[name], name))
        for name in name_to_conditioned_variational_variable:
            if name != discrete_name:
                value = E_log_prob_wrt_discrete(name_to_conditioned_variational_variable[name], name_to_substituted_value[name], name)
                tf.summary.scalar('Q(z_' + name + ')', value)

        if discrete_name is not None:
            discrete_z_probs = tf.exp(lifted_log_prob(
                name_to_unconditioned_generative_variable[discrete_name],
                name_to_substituted_value[discrete_name],  # this is the discrete states tiled over sample-index and iid-index
                discrete_name
            ))  # indexed by discrete-index, sample-index * iid-index; this is the prior (unconditioned gen.) probability of the discrete states; it will be constant over sample-index and iid-index iff the discrete has no gen. ancestors
            discrete_z_probs = tf.reduce_mean(discrete_z_probs, axis=1)  # indexed by discrete-index
            discrete_qz_probs = tf.reduce_mean(discrete_qz_probs, axis=1)  # ditto; the mean here is calculating the aggregated posterior over the batch and samples
            discrete_divergence_loss = tf.reduce_mean(tf.abs(discrete_z_probs - discrete_qz_probs)) * alpha  # L1 loss
        else:
            discrete_divergence_loss = 0.

        tf.losses.add_loss(0.)  # this is needed because get_total_loss throws instead of returning zero if no losses have been registered
        additional_losses = tf.losses.get_total_loss()
        loss = -(log_Px + (log_Pz - log_Qz) * beta) + discrete_divergence_loss + additional_losses

    tf.summary.scalar('inference/loss', loss)
    tf.summary.scalar('inference/log_Px', log_Px)
    tf.summary.scalar('inference/log_Pz', log_Pz)
    tf.summary.scalar('inference/log_Qz', log_Qz)
    tf.summary.scalar('inference/Ldd', discrete_divergence_loss)
    tf.summary.scalar('inference/L*', additional_losses)

    var_list = tf.trainable_variables()
    grads = tf.gradients(loss, [v._ref() for v in var_list])

    abs_grads = tf.abs(tf.concat([tf.reshape(grad, [-1]) for grad in grads if grad is not None], axis=0))
    loss = tf.Print(loss, [log_Px, log_Pz * beta, log_Qz * beta, discrete_divergence_loss, additional_losses, tf.reduce_mean(abs_grads), tf.reduce_max(abs_grads)], 'p(x), p(z), q(z), Ldd, L*, <|g|>, max |g| = ')

    if grad_clip_magnitude is not None:
        grads, _ = tf.clip_by_global_norm(grads, grad_clip_magnitude)

    return loss, list(zip(grads, var_list)), unconditioned_generative, reconstruction_modes, conditioned_generative

