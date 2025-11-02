# основы python №1
def count_in(text):
    check = 'aeiou'
    count = 0
    for char in text.lower():
        if char in check:
            count += 1
    return count

def tests_first():
    assert count_in("hello") == 2, "Тест 1 не пройден"
    assert count_in("bye") == 1, "Тест 2 не пройден"
    assert count_in("open the door") == 5, "Тест 3 не пройден"
    assert count_in("bcdfg") == 0, "Тест 4 не пройден"
    print('Все тесты 1 задания пройдены')

tests_first()

# основы python №2
def only_unique_chars(text):
    used_chars = set()
    for char in text:
        if char in used_chars:
            return False
        used_chars.add(char)
    return True

def tests_second():
    assert only_unique_chars("asdzxcqwe") == True, "Тест 1 не пройден"
    assert only_unique_chars("hello") == False, "Тест 2 не пройден"
    assert only_unique_chars("") == True, "Тест 3 не пройден"
    assert only_unique_chars("a") == True, "Тест 4 не пройден"
    assert only_unique_chars("aa") == False, "Тест 5 не пройден"
    print('Все тесты 2 задания пройдены')

tests_second()

# основы python №3
def count_bits(number):
    if number < 0:
        raise ValueError("Число должно быть положительным")
    return bin(number).count("1")

def tests_third():
    assert count_bits(0) == 0, "Тест 1 не пройден"
    assert count_bits(1) == 1, "Тест 2 не пройден"
    assert count_bits(2) == 1, "Тест 3 не пройден"
    assert count_bits(3) == 2, "Тест 4 не пройден"
    assert count_bits(1024) == 1, "Тест 5 не пройден"
    print('Все тесты 3 задания пройдены')

tests_third()

# основы python №4
def magic(n):
    if n < 0:
        raise ValueError("Число должно быть положительным")
    if n < 10:
        return 0
    steps = 0
    current = n
    while current >= 10:
        digits = [int(d) for d in str(current)]
        product = 1
        for digit in digits:
            product *= digit
        current = product
        steps += 1
    return steps

def tests_fourth():
    assert magic(6) == 0, "Тест 1 не пройден"
    assert magic(39) == 3, "Тест 2 не пройден"
    assert magic(77) == 4, "Тест 3 не пройден"
    assert magic(100) == 1, "Тест 4 не пройден"
    print('Все тесты 4 задания пройдены')

tests_fourth()

# основы python №5
def mse(pred, true):
    if len(pred) != len(true):
        raise ValueError("Векторы должны быть одной длины")
    if len(pred) == 0:
        raise ValueError("Векторы не могут быть пустыми")
    n = len(pred)
    sum_squared_errors = 0
    for i in range(n):
        error = pred[i] - true[i]
        sum_squared_errors += error ** 2
    mean_squared_error = sum_squared_errors / n
    return mean_squared_error

def tests_fifth():
    assert mse([1, 2, 3], [1, 2, 3]) == 0, "Тест 1 не пройден"
    assert mse([1, 2, 3], [3, 4, 5]) == 4, "Тест 2 не пройден"
    assert mse([2, 5, 7], [4, 3, 9]) == 4, "Тест 3 не пройден"
    assert mse([10], [15]) == 25, "Тест 4 не пройден"
    assert mse([-2, -3], [1, 2]) == 17, "Тест 5 не пройден"
    assert mse([1.5, 2.5], [2.0, 1.5]) == 0.625, "Тест 6 не пройден"
    print('Все тесты 5 задания пройдены')

tests_fifth()

# основы python №6
def simple_multipliers(number):
    if number < 2:
        raise ValueError("Число должно быть больше 1")
    factors = {}
    temp = number
    divisor = 2
    while divisor * divisor <= temp:
        while temp % divisor == 0:
            factors[divisor] = factors.get(divisor, 0) + 1
            temp //= divisor
        divisor += 1
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    result_parts = []
    for factor in sorted(factors.keys()):
        power = factors[factor]
        if power == 1:
            result_parts.append(f"({factor})")
        else:
            result_parts.append(f"({factor}**{power})")
    return "".join(result_parts)

def tests_sixth():
    assert simple_multipliers(8) == "(2**3)", "Тест 1 не пройден"
    assert simple_multipliers(10) == "(2)(5)", "Тест 2 не пройден"
    assert simple_multipliers(100) == "(2**2)(5**2)", "Тест 3 не пройден"
    assert simple_multipliers(15) == "(3)(5)", "Тест 4 не пройден"
    print('Все тесты 6 задания пройдены')

tests_sixth()

# основы python №7
def pyramid(number):
    total = 0
    k = 0
    while total < number:
        k += 1
        total += k * k
        if total == number:
            return k
    return "It is impossible"

def tests_seventh():
    assert pyramid(1) == 1, "Тест 1 не пройден"
    assert pyramid(5) == 2, "Тест 2 не пройден"
    assert pyramid(14) == 3, "Тест 3 не пройден"
    assert pyramid(30) == 4, "Тест 4 не пройден"
    assert pyramid(55) == 5, "Тест 5 не пройден"
    assert pyramid(2) == "It is impossible", "Тест 6 не пройден"
    assert pyramid(3) == "It is impossible", "Тест 7 не пройден"
    assert pyramid(10) == "It is impossible", "Тест 8 не пройден"
    print('Все тесты 7 задания пройдены')

tests_seventh()

# основы python №8
def is_balanced(number):
    digits = str(number)
    n = len(digits)
    if n % 2 == 0:
        mid = n // 2
        left_sum = sum(int(d) for d in digits[:mid])
        right_sum = sum(int(d) for d in digits[mid:])
    else:
        mid = n // 2
        left_sum = sum(int(d) for d in digits[:mid])
        right_sum = sum(int(d) for d in digits[mid + 1:])
    return left_sum == right_sum

def tests_eighth():
    assert is_balanced(123321) == True, "Тест 1 не пройден"
    assert is_balanced(12321) == True, "Тест 2 не пройден"
    assert is_balanced(12345) == False, "Тест 3 не пройден"
    assert is_balanced(1111) == True, "Тест 4 не пройден"
    assert is_balanced(7) == True, "Тест 5 не пройден"
    assert is_balanced(1221) == True, "Тест 6 не пройден"
    assert is_balanced(123) == False, "Тест 7 не пройден"
    print('Все тесты 8 задания пройдены')


tests_eighth()

