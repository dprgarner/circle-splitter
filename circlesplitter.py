from os import sys
from itertools import combinations


def get_lines_from_stdin():
    try:
        while True:
            yield input()
    except EOFError:
        pass


def get_lines_from_file(input_file):
    with open(input_file, 'r') as in_:
        while True:
            line = in_.readline().replace('\n', '')
            if not line:
                break
            yield line


INFINITY = 'infinity'
EPSILON = pow(10, -8)  # To account for rounding errors.
DECIMAL_PLACES = 10


class CircleSplitterParser(object):
    """
    An attempt at solving this circle-splitting problem:
    https://www.reddit.com/r/dailyprogrammer/comments/6ksmh5/20170630_challenge_321_hard_circle_splitter/

    Use this class by initialising at the top level, and either piping to
    stdin/stdout, or specifying the input and output in the command-line
    arguments.
    """
    def __init__(self, plot=False):
        self.plot = plot
        self.source = get_lines_from_stdin()
        if len(sys.argv) > 1:
            self.source = get_lines_from_file(sys.argv[1])

        if len(sys.argv) > 2:
            with open(sys.argv[2], 'w') as out_:
                self.dest = out_
                self.write_output()
        else:
            self.dest = sys.stdout
            self.write_output()

    def write_output(self):
        n = int(next(self.source))
        points = []
        for i in range(n):
            x, y = map(float, next(self.source).split(' '))
            points.append((x, y))

        output = self.solve(points)
        if output:
            x, y, r = output
            self.dest.write('{} {}\n{}'.format(
                round(x, DECIMAL_PLACES),
                round(y, DECIMAL_PLACES),
                round(r, DECIMAL_PLACES),
            ))
        else:
            self.dest.write('No solution')

    def debug(self, points, circle):
        """
        Plot this with matplotlib to see if it looks right.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig = plt.figure(0)
        axes = fig.add_subplot(111, aspect='equal')

        x, y, r = circle
        circle = Circle((x, y), r)

        axes.add_artist(circle)

        circle.set_clip_box(axes.bbox)
        circle.set_facecolor('none')
        circle.set_edgecolor([0, 0, 0])

        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)

        xarr = np.zeros(len(points))
        yarr = np.zeros(len(points))

        for i, (x, y) in enumerate(points):
            xarr[i] = x
            yarr[i] = y

        plt.scatter(xarr, yarr, marker='X')
        plt.show()

    def perpendicular_bisection(self, p1, p2):
        """
        Find the perpendicular bisection of the line which intersects the two
        points. Returns (m, c) if the line is of the form y = mx + c, or
        (INFINITY, c) if the line is of the form x = c.
        """
        x1, y1 = p1
        x2, y2 = p2
        if y2 == y1:
            return (INFINITY, (x1 + x2) / 2.0)

        m = (x1 - x2) / (y2 - y1)
        c = (y2 * y2 - y1 * y1 + x2 * x2 - x1 * x1) / (2.0 * (y2 - y1))
        return (m, c)

    def line_intersection(self, l1, l2):
        """
        Given a pair of lines, returns the intersection point if it exists or
        None if the lines are parallel.
        """
        m1, c1 = l1
        m2, c2 = l2

        if m1 == m2:
            return None
        elif m1 == INFINITY:
            return (c1, m2 * c1 + c2)
        elif m2 == INFINITY:
            return (c2, m1 * c2 + c1)
        else:
            x = (c2 - c1) / float(m1 - m2)
            y = (m1 * c2 - m2 * c1) / float(m1 - m2)
            return (x, y)

    def circumscribe(self, p1, p2, p3):
        """
        Given the points p1, p2, p3, return the circumcentre and the radius of
        the circle, or None if this does not exist.
        """
        l1 = self.perpendicular_bisection(p1, p2)
        l2 = self.perpendicular_bisection(p1, p3)
        p = self.line_intersection(l1, l2)

        if not p:
            return None

        x, y = p
        x1, y1 = p1
        r = pow(pow(x - x1, 2) + pow(y - y1, 2), 0.5)

        return (*p, r)

    def bounded(self, circle):
        x0, y0, r = circle
        if x0 - r < 0:
            return False
        if x0 + r > 1:
            return False
        if y0 - r < 0:
            return False
        if y0 + r > 1:
            return False
        return True

    def circles(self, points):
        for p1, p2, p3 in combinations(points, 3):
            if p1 == p2 or p2 == p3 or p1 == p3:
                continue
            circle = self.circumscribe(p1, p2, p3)
            if circle and self.bounded(circle):
                yield circle

        for p1, p2 in combinations(points, 2):
            if p1 == p2:
                continue
            (x1, y1), (x2, y2) = p1, p2
            r = pow(pow((x1 - x2) / 2.0, 2) + pow((y1 - y2) / 2.0, 2), 0.5)
            x0 = (x1 + x2) / 2.0
            y0 = (y1 + y2) / 2.0
            circle = (x0, y0, r)
            if self.bounded(circle):
                yield circle

    def solve(self, points):
        """
        Return (x, y, r) if there is a solution, or None if not.
        """
        n = len(points)

        if n == 2:
            return points[0]

        min_r = 1
        min_x, min_y = None, None

        # Find all triples of points in which the circumcircle contains
        # exactly half of the points, and also the circles intersecting
        # exactly two points.
        for circle in self.circles(points):
            x0, y0, r = circle
            r_squared = pow(r, 2)

            # Count the number of points in the circle.
            count = 0
            for x, y in points:
                if pow(x - x0, 2) + pow(y - y0, 2) <= r_squared + EPSILON:
                    count += 1

            if 2 * count == n and r < min_r:
                min_r = r
                min_x = x0
                min_y = y0

        # Any solutions found?
        if min_x == None:
            return

        # Should we plot the circle?
        if self.plot:
            self.debug(points, (min_x, min_y, min_r))

        return (min_x, min_y, min_r)


CircleSplitterParser()
