def __directions_to_str(directions):
    st_all = []
    for direction in directions:
        st = ""
        for i in direction:
            st += str(int(i[0] * 4))
        st_all.append(st)
    return st_all


def check_overlap(real_directions, fake_directions):
    real_st = __directions_to_str(real_directions)
    fake_st = __directions_to_str(fake_directions)

    overlap = 0
    for i in fake_st:
        if i in real_st:
            overlap += 1
    return overlap


def check_unique(fake_directions):
    fake_st = __directions_to_str(fake_directions)
    unique_st = list(set(fake_st))
    num_unique = len(unique_st)
    return num_unique
