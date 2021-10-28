from .GIMME import _GIMME_follow_up, _GIMME_post_process
from .Eflux import _Eflux_follow_up


# Follow up functions will give users further info about the constraints, they are not included in the analysis df
def follow_up(constraint_details, constr, **kwargs):
    if constr == "Eflux":
        return _Eflux_follow_up(constraint_details, **kwargs)
    if constr == "GIMME":
        return _GIMME_follow_up(constraint_details, **kwargs)


def post_process(sol_df, constr, **kwargs):
    if constr == "GIMME":
        return _GIMME_post_process(sol_df, **kwargs)

    print("No post processing method found, an identical frame was returned")
    return sol_df