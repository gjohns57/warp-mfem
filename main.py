import warp as wp


def main():
    x = 10.0
    msg = "hello"

    def inner_func():
        nonlocal x
        x -= 1.0
        print(msg)

    #
    # inner_function = lambda _: x -= 1.0

    while x >= 0.0:
        inner_func()


if __name__ == "__main__":
    main()
