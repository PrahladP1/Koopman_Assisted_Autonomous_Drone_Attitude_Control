import numpy as np

def compute_pos_err(x, x_ref):
    return np.linalg.norm(x[0:3 - x_ref[0:3]])

def compute_vel_err(x, x_ref):
    return np.linalg.norm(x[3:6] - x_ref[3:6])

def compute_att_err(x, x_ref):
    return np.linalg.norm(x[6:9] - x_ref[6:9])

def compute_body_err(x, x_ref):
    return np.linalg.norm(x[9:12] - x_ref[9:12])

def compute_sum_err(x, x_ref):
    return np.linalg.norm(x - x_ref)

def rms_err(trajectory, reference,):
    err = trajectory - reference
    return np.sqrt(np.mean(err**2))

def final_pos_err(trajectory, reference,):
    return compute_pos_err(trajectory[-1], reference[-1],)

def final_att_err(trajectory, reference,):
    return compute_att_err(trajectory[-1], reference[-1],)

def peak_pos_err(trajectory, reference,):
    errs = np.linalg.norm(trajectory[:, 0:3] - reference[:, 0:3], axis=1)
    return np.max(errs)

def integrated_abs_err(trajectory, reference, dt,):
    errs = np.linalg.norm(trajectory[:, 0:3] - reference[:, 0:3], axis=1)
    return np.sum(errs)*dt

def control_effort(controls,):
    return np.sum(controls**2)

def max_control(controls,):
    return np.max(np.abs(controls))

def saturate_percentage(controls, limit,):
    sat = np.abs(controls) >= limit
    return np.mean(sat)

def success(pos_err, threshold,):
    return pos_err < threshold

def compute_all_metrics(trajectory, reference, controls, dt,):
    return {"final_pos_err": final_pos_err(trajectory, reference,),
            "final_att_err": final_att_err(trajectory, reference,),
            "peak_position_err": peak_pos_err(trajectory, reference,),
            "rms_err": rms_err(trajectory, reference,),
            "iae": integrated_abs_err(trajectory, reference, dt,),
            "control_effort": control_effort(controls,),
            "max_control": max_control(controls,),}