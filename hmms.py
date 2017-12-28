import numpy as np
np.set_printoptions(threshold=np.nan)

def forward(x, pi, A, B):
    """ Run the forward algorithm for a single example.

    Args:
        x: A 1-D int NumPy array with shape [T], where each element
            is either 0, 1, 2, ..., or N_x - 1. T is the length of
            the observation sequence and N_x is the number of possible
            values that each observation can take on.
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on.
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i).
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i).

    Returns:
        alpha, a 2-D float NumPy array with shape [T, N_z].
    """
    # TODO: Write this function.
    #x = x[1]
    B_col = B[:, x[0]] # [N_z, 1]
    alpha = np.multiply(pi, B_col)
    ret = np.zeros((x.shape[0], pi.shape[0]))
    ret[0] = alpha
    for i in range(1, x.shape[0]):
        B_col = B[:, x[i]]
        sum_term = np.dot(A, alpha) #before: alpha, A
        alpha = np.multiply(B_col, sum_term) #before: sum_term before
        ret[i] = alpha
    return ret

def backward(x, pi, A, B):
    """ Run the backward algorithm for a single example.

    Args:
        x: A 1-D int NumPy array with shape [T], where each element
            is either 0, 1, 2, ..., or N_x - 1. T is the length of
            the observation sequence and N_x is the number of possible
            values that each observation can take on.
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on.
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i).
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i).

    Returns:
        beta, a 2-D float NumPy array with shape [T, N_z].
    """
    # TODO: Write this function.
    #Beta_index = np.ones(2)
    #ret = np.zeros((x.shape[0], pi.shape[0]))
    #ret[x.shape[0]-1] = Beta_index
    #for i in range(x.shape[0]-2, -1, -1):
    #    B_col = B[:, x[i+1]]
    #    sum_term = np.dot(B_col, A)
    #    Beta_index = np.multiply(sum_term, Beta_index)
    #    ret[i] = Beta_index
    #return ret
    Beta_index = np.ones(2)
    ret = np.zeros((x.shape[0], pi.shape[0]))
    ret[x.shape[0]-1] = Beta_index
    for i in range(x.shape[0]-2, -1, -1):
        B_col = B[:, x[i+1]]
        sum_term = np.multiply(Beta_index, B_col)
        Beta_index = np.dot(A, sum_term)
        ret[i] = Beta_index
    return ret



def individually_most_likely_states(X, pi, A, B):
    """ Computes individually most-likely states.

    By "individually most-likely states," we mean that the *marginal*
    distributions are maximized. In other words, for any particular
    time step of any particular sequence, each returned state i is
    chosen to maximize P(z_t = i | x).

    All sequences in X are assumed to have the same length, T.

    Args:
        X: A 2-D int NumPy array with shape [N, T], where each element
            is either 0, 1, 2, ..., or N_x - 1. N is the number of observation
            sequences, T is the length of every sequence, and N_x is the number
            of possible values that each observation can take on. [50 30] of 0, 1's
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on. [.8 .2]
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i). [2 by 2]
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i). [2 by 2]

    Returns:
        Z, a 2-D int NumPy array with shape [N, T], where each element
            is either 0, 1, 2, ..., N_z - 1. [50 by 30]
    """
    # TODO: Write this function.
    Z = np.zeros((X.shape[0], X.shape[1]), dtype=np.int)
    for i in range(X.shape[0]):
        alpha = forward(X[i], pi, A, B) # [T, N_z] 30, 2
        beta = backward(X[i], pi, A, B)
        p_x = np.sum(alpha[X.shape[1]-1]) #
        p_z_t = np.multiply(alpha, beta) / p_x
        temp = np.argmax(p_z_t, axis=1)
        Z[i] = temp
    return Z


def take_EM_step(X, pi, A, B):
    """ Take a single expectation-maximization step.

    Args:
        X: A 2-D int NumPy array with shape [N, T], where each element
            is either 0, 1, 2, ..., or N_x - 1. N is the number of observation
            sequences, T is the length of every sequence, and N_x is the number
            of possible values that each observation can take on.
        pi: A 1-D float NumPy array with shape [N_z]. N_z is the number
            of possible values that each hidden state can take on.
        A: A 2-D float NumPy array with shape [N_z, N_z]. A[i, j] is
            the probability of transitioning from state i to state j:
            A[i, j] = P(z_t = j | z_t-1 = i).
        B: A 2-D float NumPy array with shape [N_z, N_x]. B[i, j] is
            the probability of from state i emitting observation j:
            B[i, j] = P(x_t = j | z_t = i).

    Returns:
        A tuple containing
        pi_prime: pi after the EM update.
        A_prime: A after the EM update.
        B_prime: B after the EM update.
    """
    # TODO: Write this function.
    pi_prime = np.zeros(pi.shape[0])
    A_prime = np.zeros((A.shape[0], A.shape[1]))
    B_prime = np.zeros((B.shape[0], B.shape[1]))
    for i in range(X.shape[0]):
        alpha = forward(X[i], pi, A, B)
        beta = backward(X[i], pi, A, B)
        p_xn = np.sum(alpha[X.shape[1]-1])

        #pi update
        alpha_0 = alpha[0] # alpha_0, i
        beta_0 = beta[0]
        pi_update = np.multiply(alpha_0, beta_0) / p_xn
        pi_prime += pi_update

        #a update
        for i_val in range(A.shape[0]):
            for j_val in range(A.shape[1]):
                tem = 0
                for t_val in range(X.shape[1]-1):
                    tem += alpha[t_val][i_val] * A[i_val][j_val] * B[j_val][X[i][t_val+1]] * beta[t_val+1][j_val]
                tem = tem / p_xn
                A_prime[i_val][j_val] += tem

        #b update
        for k in range(B.shape[1]):
            b_sum = np.zeros(2)
            for p in range(X.shape[1]):
                if X[i][p] == k:
                    alpha_t_i = alpha[p]
                    beta_t_i = beta[p]
                    b_sum += np.multiply(alpha_t_i, beta_t_i)
            b_sum = b_sum / p_xn
            B_prime[:, k] += b_sum

    #normalization
    pi_prime = pi_prime/pi_prime.sum()
    for e in range(A_prime.shape[0]):
        A_prime[e] = A_prime[e]/A_prime[e].sum()
    for d in range(B_prime.shape[0]):
        B_prime[d] = B_prime[d]/B_prime[d].sum()
    return (pi_prime, A_prime, B_prime)




