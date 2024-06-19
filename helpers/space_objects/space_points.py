def space_points(points, spacing=1, iterations=5):
    """Function to intelligently space out a list of points
    such that they are as close to their original positions
    as possible while maintaining a certain buffer from one
    another

    These points are on a linear scale
    """

    def space(group, spacing):
        """Function to evenly space points around the
        centroid of the original group"""
        center = sum(group) / len(group)  # determine center of group
        final_width = (len(group) - 1) * spacing
        beg = center - final_width / 2
        return [beg + i * spacing for i in range(len(group))]

    def clump(points, spacing):
        """Function to group points together based on
        their proximity to one another"""
        points.sort()
        groups = [[pt] for pt in points]
        clumped_groups = []
        cand = groups.pop(0)
        while groups:
            spaced_cand = space(cand, spacing)
            if spaced_cand[-1] + spacing > groups[0][0]:
                cand += groups.pop(0)
            else:
                clumped_groups.append(cand)
                cand = groups.pop(0)
        clumped_groups.append(cand)
        return clumped_groups

    for _ in range(iterations):
        clumps = clump(points, spacing)
        result = []
        for item in clumps:
            result += space(item, spacing)
        points = result
    return result
