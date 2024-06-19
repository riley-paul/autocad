def format_KP(number, comma=False) -> str:
    """Formats chainage number as '#+###' e.g. 34032.43 to 34+032"""
    if type(number) == int or float:
        post_plus = number % 1000
        pre_plus = (number - post_plus) / 1000
        return (
            f"{pre_plus:,.0f}+{post_plus:03.0f}"
            if comma
            else f"{pre_plus:.0f}+{post_plus:03.0f}"
        )
    else:
        return number
