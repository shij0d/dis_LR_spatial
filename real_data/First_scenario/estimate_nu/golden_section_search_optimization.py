import math

# Define the Golden Ratio constant (tau, or 1/phi)
# r = 2 - phi = 2 - (1 + sqrt(5))/2 = (3 - sqrt(5))/2 approx 0.381966
# Or, more simply, use the common ratio:
R_GOLDEN = (math.sqrt(5) - 1) / 2 # ~ 0.618034 (1/phi)
r = 1 - R_GOLDEN # ~ 0.381966 (1/phi^2)

def golden_section_search_optimization(func, L, R, epsilon=1e-6):
    """
    Optimizes (finds the minimum) of a unimodal function on [L, R] 
    using the Golden Section Search method. This is more efficient than 
    Ternary Search because it reuses one function evaluation per step.

    Args:
        func (function): The 1D function to be minimized, f(x).
        L (float): The left boundary of the search interval.
        R (float): The right boundary of the search interval.
        epsilon (float): The desired tolerance for the final interval length.

    Returns:
        float: The x-value where the function is minimized.
    """
    
    # 1. Initialize the internal points and function values
    # Calculate the first two points m1 and m2 using the Golden Ratio
    m1 = R - R_GOLDEN * (R - L) # m1 is further from L
    m2 = L + R_GOLDEN * (R - L) # m2 is further from R
    
    f_m1 = func(m1)
    f_m2 = func(m2)
    
    # 
    
    # 2. Iteration Loop
    while (R - L) > epsilon:
        # Compare function values (looking for the minimum)
        if f_m1 < f_m2:
            # Minimum is in [L, m2]. Discard the rightmost section (m2 to R).
            R = m2
            
            # The old m1 becomes the new m2 (reuse the value f_m1)
            m2 = m1
            f_m2 = f_m1
            
            # Calculate the NEW m1 and its function value
            m1 = R - R_GOLDEN * (R - L)
            f_m1 = func(m1)
            
        else: # f_m1 >= f_m2
            # Minimum is in [m1, R]. Discard the leftmost section (L to m1).
            L = m1
            
            # The old m2 becomes the new m1 (reuse the value f_m2)
            m1 = m2
            f_m1 = f_m2
            
            # Calculate the NEW m2 and its function value
            m2 = L + R_GOLDEN * (R - L)
            f_m2 = func(m2)
            
    # 3. Termination: Return the midpoint of the final interval
    return (L + R) / 2