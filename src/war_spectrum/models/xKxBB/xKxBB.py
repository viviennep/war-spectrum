import numpy as np, polars as pl, joblib
cl = pl.col

def markov_pa_outcome_probs(P):
    '''
    Absorbing markov transition matrix (notation from wikipedia):
        Π = [Q R]
            [0 I]
    • Where Q encodes transitions from transient to transient states (counts)
      and R encodes transitions from transient states to absorbing states (pa outcomes)
    • There are 4 ball states*3 strike states = 12 transient states
      and 4 PA outcomes (that we care about here): K, BB, HBP, BIP
    • The probability of being absorbed by state j when starting in state i is given by
      the i,j entry of (I-Q)^-1 @ R
    So, loop through all transient states and fill up Q and R based on pitch outcome probs
    solve for (I-Q)^-1 @ R and take the first row for per-PA outcome probs
    '''
    Q = np.zeros((4,3,4,3))
    R = np.zeros((4,3,4))
    K,BB,HBP,BIP = np.arange(4)
    for b in range(4):
        for s in range(3):
            p_ball, p_called_strike, p_whiff, p_foul, p_hbp, p_bip = P[b,s]
            # Balls
            if b<3:
                Q[b,s,b+1,s] += p_ball
            else:
                R[b,s,BB] += p_ball
            # Strikes
            for p_strike in (p_whiff,p_called_strike):
                if s<2:
                    Q[b,s,b,s+1] += p_strike
                else:
                    R[b,s,K] += p_strike
            # Fouls
            if s<2:
                Q[b,s,b,s+1] += p_foul
            else:
                Q[b,s,b,s] += p_foul
            # HBP
            R[b,s,HBP] += p_hbp
            # BIP
            R[b,s,BIP] += p_bip
    Q = Q.reshape(12,12)
    R = R.reshape(12,4)
    return (np.linalg.inv(np.eye(12)-Q)@R)[0]
'''
Absorbing markov transition matrix (notation from wikipedia):
    Π = [Q R]
        [0 I]
• Where Q encodes transitions from transient to transient states (counts)
  and R encodes transitions from transient states to absorbing states (pa outcomes)
• There are 4 ball states*3 strike states = 12 transient states
  and 4 PA outcomes (that we care about here): K, BB, HBP, BIP
• The probability of being absorbed by state j when starting in state i is given by
  the i,j entry of (I-Q)^-1 @ R
So, loop through all transient states and fill up Q and R based on pitch outcome probs
solve for (I-Q)^-1 @ R and take the first row for per-PA outcome probs
'''

