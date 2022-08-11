from .GIMME import _GIMME_follow_up, _GIMME_post_process


# Follow up functions will give users further info about the constraints, they are not included in the analysis df
def follow_up(constraint_details, constr, **kwargs):
    if constr == "GIMME":
        return _GIMME_follow_up(constraint_details, **kwargs)
    if constr == "RIPTiDe":
        return _GIMME_follow_up(constraint_details, **kwargs)


def post_process(sol_df, constr, **kwargs):
    if constr == "GIMME":
        return _GIMME_post_process(sol_df, **kwargs)
    if constr == "RIPTiDe":
        return _GIMME_post_process(sol_df, **kwargs)
    return sol_df