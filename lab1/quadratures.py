def rectangles(f, a, b, n):
    h = (b - a) / n
    s = 0

    for i in range(1, n + 1):
        s += f(a + h * i - h / 2)

    return s * h


def trapezoids(f, a, b, n):
    h = (b - a) / n
    s = (f(a) + f(b)) / 2

    for i in range(1, n):
        s += f(a + h * i)

    return s * h


def simpson(f, a, b, n):
    h = (b - a) / n
    s = f(a) + f(b)

    for i in range(1, n):
        if i % 2 != 0:
            s += 4 * f(a + h * i)
        else:
            s += 2 * f(a + h * i)

    return h / 3 * s


def three_eights(f, a, b, n):
    h = (b - a) / n
    s = f(a) + f(b)

    for i in range(1, n):
        if i % 3 == 0:
            s += 2 * f(a + h * i)
        else:
            s += 3 * f(a + h * i)

    return h * 3 / 8 * s
