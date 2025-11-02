import numpy as np
import matplotlib.pyplot as plt

# numpy №1
def sum_prod(X, V):
    result = np.zeros_like(V[0])
    for i in range(len(X)):
        result += X[i] @ V[i]
    return result


def tests_numpy_first():
    X1 = [np.array([[1, 2], [3, 4]])]
    V1 = [np.array([[1], [2]])]
    expected1 = np.array([[5], [11]])
    assert np.array_equal(sum_prod(X1, V1), expected1), "Тест 1 не пройден"
    X2 = [np.array([[1, 0], [0, 1]]), np.array([[2, 0], [0, 2]])]
    V2 = [np.array([[1], [1]]), np.array([[2], [2]])]
    expected2 = np.array(
        [[5], [5]])
    assert np.array_equal(sum_prod(X2, V2), expected2), "Тест 2 не пройден"
    X3 = [np.array([[2]]), np.array([[3]]), np.array([[4]])]
    V3 = [np.array([[5]]), np.array([[6]]), np.array([[7]])]
    expected3 = np.array([[56]])
    assert np.array_equal(sum_prod(X3, V3), expected3), "Тест 3 не пройден"
    X4 = [np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]])]
    V4 = [np.array([[1], [2]]), np.array([[3], [4]])]
    expected4 = np.array([[0], [0]])
    assert np.array_equal(sum_prod(X4, V4), expected4), "Тест 4 не пройден"
    print('Все тесты numpy 1 задания пройдены')

tests_numpy_first()

# numpy №2
def binarize(M, threshold=0.5):
    return (M > threshold).astype(int)


def tests_numpy_second():
    M1 = np.array([[0.1, 0.6], [0.4, 0.9]])
    expected1 = np.array([[0, 1], [0, 1]])
    assert np.array_equal(binarize(M1), expected1), "Тест 1 не пройден"
    M2 = np.array([[0.3, 0.7], [0.5, 0.8]])
    expected2 = np.array([[0, 1], [0, 1]])
    assert np.array_equal(binarize(M2, 0.5), expected2), "Тест 2 не пройден"
    M3 = np.array([[0.6, 0.8], [0.7, 0.9]])
    expected3 = np.array([[0, 1], [0, 1]])
    assert np.array_equal(binarize(M3, 0.7), expected3), "Тест 3 не пройден"
    M4 = np.array([[0.1, 0.2], [0.3, 0.4]])
    expected4 = np.array([[0, 0], [0, 0]])
    assert np.array_equal(binarize(M4, 0.5), expected4), "Тест 4 не пройден"
    M5 = np.array([[0.6, 0.7], [0.8, 0.9]])
    expected5 = np.array([[1, 1], [1, 1]])
    assert np.array_equal(binarize(M5, 0.5), expected5), "Тест 5 не пройден"
    M6 = np.array([[-1.0, 0.0], [0.5, 1.0]])
    expected6 = np.array([[0, 0], [0, 1]])
    assert np.array_equal(binarize(M6, 0.5), expected6), "Тест 6 не пройден"
    print('Все тесты numpy 2 задания пройдены')

tests_numpy_second()

# numpy №3
def unique_rows(mat):
    result = []
    for i in range(mat.shape[0]):
        result.append(np.unique(mat[i]))
    return result


def unique_columns(mat):
    result = []
    for j in range(mat.shape[1]):
        result.append(np.unique(mat[:, j]))
    return result


def tests_numpy_third():
    mat1 = np.array([[1, 2, 3],
                     [2, 2, 1],
                     [3, 3, 3]])
    rows_result = unique_rows(mat1)
    expected_rows = [np.array([1, 2, 3]), np.array([1, 2]), np.array([3])]
    for i in range(len(rows_result)):
        assert np.array_equal(rows_result[i], expected_rows[i]), f"Тест 1 rows не пройден, строка {i}"
    cols_result = unique_columns(mat1)
    expected_cols = [np.array([1, 2, 3]), np.array([2, 3]), np.array([1, 3])]
    for i in range(len(cols_result)):
        assert np.array_equal(cols_result[i], expected_cols[i]), f"Тест 1 cols не пройден, столбец {i}"
    mat2 = np.array([[1, 2],
                     [3, 4]])
    rows_result2 = unique_rows(mat2)
    expected_rows2 = [np.array([1, 2]), np.array([3, 4])]
    for i in range(len(rows_result2)):
        assert np.array_equal(rows_result2[i], expected_rows2[i]), f"Тест 2 rows не пройден, строка {i}"
    cols_result2 = unique_columns(mat2)
    expected_cols2 = [np.array([1, 3]), np.array([2, 4])]
    for i in range(len(cols_result2)):
        assert np.array_equal(cols_result2[i], expected_cols2[i]), f"Тест 2 cols не пройден, столбец {i}"
    mat3 = np.array([[1, 1, 1],
                     [2, 2, 2]])
    rows_result3 = unique_rows(mat3)
    expected_rows3 = [np.array([1]), np.array([2])]
    for i in range(len(rows_result3)):
        assert np.array_equal(rows_result3[i], expected_rows3[i]), f"Тест 3 rows не пройден, строка {i}"
    cols_result3 = unique_columns(mat3)
    expected_cols3 = [np.array([1, 2]), np.array([1, 2]), np.array([1, 2])]
    for i in range(len(cols_result3)):
        assert np.array_equal(cols_result3[i], expected_cols3[i]), f"Тест 3 cols не пройден, столбец {i}"
    mat4 = np.array([[1, 2, 1, 3, 2]])
    rows_result4 = unique_rows(mat4)
    expected_rows4 = [np.array([1, 2, 3])]
    assert np.array_equal(rows_result4[0], expected_rows4[0]), "Тест 4 rows не пройден"
    cols_result4 = unique_columns(mat4)
    expected_cols4 = [np.array([1]), np.array([2]), np.array([1]), np.array([3]), np.array([2])]
    for i in range(len(cols_result4)):
        assert np.array_equal(cols_result4[i], expected_cols4[i]), f"Тест 4 cols не пройден, столбец {i}"
    print('Все тесты numpy 3 задания пройдены')

tests_numpy_third()

# numpy №4
def analyze_matrix(m, n, mean=0, std=1):
    matrix = np.random.normal(mean, std, size=(m, n))
    row_means = np.mean(matrix, axis=1)
    row_vars = np.var(matrix, axis=1)
    col_means = np.mean(matrix, axis=0)
    col_vars = np.var(matrix, axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for i in range(min(m, 5)):
        axes[0, 0].hist(matrix[i], alpha=0.6, label=f'Строка {i + 1}', bins=15)
    axes[0, 0].set_title('Гистограммы значений по строкам')
    axes[0, 0].set_xlabel('Значения')
    axes[0, 0].set_ylabel('Частота')
    axes[0, 0].legend()
    for j in range(min(n, 5)):
        axes[0, 1].hist(matrix[:, j], alpha=0.6, label=f'Столбец {j + 1}', bins=15)
    axes[0, 1].set_title('Гистограммы значений по столбцам')
    axes[0, 1].set_xlabel('Значения')
    axes[0, 1].set_ylabel('Частота')
    axes[0, 1].legend()
    axes[1, 0].bar(range(m), row_means, alpha=0.7, color='blue')
    axes[1, 0].set_title('Мат. ожидание по строкам')
    axes[1, 0].set_xlabel('Номер строки')
    axes[1, 0].set_ylabel('Мат. ожидание')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].bar(range(n), col_means, alpha=0.7, color='red')
    axes[1, 1].set_title('Мат. ожидание по столбцам')
    axes[1, 1].set_xlabel('Номер столбца')
    axes[1, 1].set_ylabel('Мат. ожидание')
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return {
        'matrix': matrix,
        'row_means': row_means,
        'row_vars': row_vars,
        'col_means': col_means,
        'col_vars': col_vars
    }


def tests_numpy_fourth():
    result = analyze_matrix(3, 4)
    assert result['matrix'].shape == (3, 4), "Тест 1 не пройден"
    assert len(result['row_means']) == 3, "Тест 2 не пройден"
    assert len(result['col_means']) == 4, "Тест 3 не пройден"
    assert len(result['row_vars']) == 3, "Тест 4 не пройден"
    assert len(result['col_vars']) == 4, "Тест 5 не пройден"
    print('Все тесты numpy 4 задания пройдены')

tests_numpy_fourth()

# numpy №5
def chess(m, n, a, b):
    matrix = np.zeros((m, n), dtype=type(a))
    for i in range(m):
        for j in range(n):
            if (i + j) % 2 == 0:
                matrix[i, j] = a
            else:
                matrix[i, j] = b
    return matrix


def tests_numpy_fifth():
    result1 = chess(2, 2, 0, 1)
    expected1 = np.array([[0, 1], [1, 0]])
    assert np.array_equal(result1, expected1), "Тест 1 не пройден"
    result2 = chess(3, 3, 'A', 'B')
    expected2 = np.array([['A', 'B', 'A'],
                          ['B', 'A', 'B'],
                          ['A', 'B', 'A']])
    assert np.array_equal(result2, expected2), "Тест 2 не пройден"
    result3 = chess(1, 1, 5, 10)
    expected3 = np.array([[5]])
    assert np.array_equal(result3, expected3), "Тест 3 не пройден"
    result4 = chess(1, 4, 7, 8)
    expected4 = np.array([[7, 8, 7, 8]])
    assert np.array_equal(result4, expected4), "Тест 4 не пройден"
    result5 = chess(4, 1, 1, 2)
    expected5 = np.array([[1], [2], [1], [2]])
    assert np.array_equal(result5, expected5), "Тест 5 не пройден"
    result6 = chess(2, 2, -1, 1)
    expected6 = np.array([[-1, 1], [1, -1]])
    assert np.array_equal(result6, expected6), "Тест 6 не пройден"
    result7 = chess(2, 3, 0.5, 1.5)
    expected7 = np.array([[0.5, 1.5, 0.5], [1.5, 0.5, 1.5]])
    assert np.array_equal(result7, expected7), "Тест 7 не пройден"
    print('Все тесты numpy 5 задания пройдены')

tests_numpy_fifth()

# numpy №6
def draw_rectangle(a, b, m, n, rectangle_color, background_color):
    image = np.full((m, n, 3), background_color, dtype=np.uint8)
    x_start = (n - a) // 2
    y_start = (m - b) // 2
    x_end = x_start + a
    y_end = y_start + b
    image[y_start:y_end, x_start:x_end] = rectangle_color
    return image


def draw_ellipse(a, b, m, n, ellipse_color, background_color):
    image = np.full((m, n, 3), background_color, dtype=np.uint8)
    x0, y0 = n // 2, m // 2
    for y in range(m):
        for x in range(n):
            if ((x - x0) ** 2) / (a ** 2) + ((y - y0) ** 2) / (b ** 2) <= 1:
                image[y, x] = ellipse_color
    return image


def tests_numpy_sixth():
    rect1 = draw_rectangle(2, 2, 4, 4, [255, 0, 0], [0, 0, 0])
    expected_center = np.array([[[255, 0, 0], [255, 0, 0]],
                                [[255, 0, 0], [255, 0, 0]]])
    assert np.array_equal(rect1[1:3, 1:3], expected_center), "Тест 1 не пройден"
    rect2 = draw_rectangle(1, 1, 3, 3, [0, 255, 0], [255, 255, 255])
    assert np.array_equal(rect2[1, 1], [0, 255, 0]), "Тест 2 не пройден"
    assert np.array_equal(rect2[0, 0], [255, 255, 255]), "Тест 3 не пройден"
    ellipse1 = draw_ellipse(1, 1, 5, 5, [0, 0, 255], [255, 255, 255])
    assert np.array_equal(ellipse1[2, 2], [0, 0, 255]), "Тест 4 не пройден"
    assert np.array_equal(ellipse1[0, 0], [255, 255, 255]), "Тест 5 не пройден"
    ellipse2 = draw_ellipse(10, 10, 5, 5, [255, 0, 0], [0, 0, 0])
    assert np.all(ellipse2 == [255, 0, 0]), "Тест 6 не пройден"
    rect3 = draw_rectangle(3, 3, 10, 10, [100, 100, 100], [200, 200, 200])
    assert rect3.shape == (10, 10, 3), "Тест 7 не пройден"
    print('Все тесты numpy 6 задания пройдены')

tests_numpy_sixth()

# numpy №7
def analyze_time_series(series, window_size):
    mean = np.mean(series)
    variance = np.var(series)
    std_dev = np.std(series)
    local_maxima = []
    local_minima = []
    for i in range(1, len(series) - 1):
        if series[i] > series[i - 1] and series[i] > series[i + 1]:
            local_maxima.append(i)
        elif series[i] < series[i - 1] and series[i] < series[i + 1]:
            local_minima.append(i)
    moving_average = []
    for i in range(len(series) - window_size + 1):
        window = series[i:i + window_size]
        moving_average.append(np.mean(window))
    return {
        'mean': mean,
        'variance': variance,
        'std_dev': std_dev,
        'local_maxima': local_maxima,
        'local_minima': local_minima,
        'moving_average': moving_average
    }


def tests_numpy_seventh():
    series1 = [1, 2, 3, 2, 1]
    result1 = analyze_time_series(series1, 3)
    assert result1['mean'] == 1.8, "Тест 1 не пройден"
    assert abs(result1['variance'] - 0.56) < 0.01, "Тест 2 не пройден"
    assert abs(result1['std_dev'] - 0.748) < 0.01, "Тест 3 не пройден"
    assert result1['local_maxima'] == [2], "Тест 4 не пройден"
    assert result1['local_minima'] == [], "Тест 5 не пройден"
    assert np.allclose(result1['moving_average'], [2.0, 2.333, 2.0]), "Тест 6 не пройден"
    print('Все тесты numpy 7 задания пройдены')

tests_numpy_seventh()

#numpy №8
def one_hot_encoding(labels):
    num_classes = len(np.unique(labels))
    one_hot = np.zeros((len(labels), num_classes), dtype=int)
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
    return one_hot


def tests_numpy_eighth():
    labels1 = [0, 2, 3, 0]
    result1 = one_hot_encoding(labels1)
    expected1 = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1],
                          [1, 0, 0, 0]])
    assert np.array_equal(result1, expected1), "Тест 1 не пройден"
    labels2 = [0, 1, 1, 0]
    result2 = one_hot_encoding(labels2)
    expected2 = np.array([[1, 0],
                          [0, 1],
                          [0, 1],
                          [1, 0]])
    assert np.array_equal(result2, expected2), "Тест 2 не пройден"
    labels3 = [0, 0, 0]
    result3 = one_hot_encoding(labels3)
    expected3 = np.array([[1],
                          [1],
                          [1]])
    assert np.array_equal(result3, expected3), "Тест 3 не пройден"
    labels4 = [1, 2, 3]
    result4 = one_hot_encoding(labels4)
    expected4 = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
    assert np.array_equal(result4, expected4), "Тест 4 не пройден"
    labels5 = [0, 5, 2]
    result5 = one_hot_encoding(labels5)
    expected5 = np.array([[1, 0, 0],
                          [0, 0, 1],
                          [0, 1, 0]])
    assert np.array_equal(result5, expected5), "Тест 5 не пройден"
    labels6 = [2]
    result6 = one_hot_encoding(labels6)
    expected6 = np.array([[0, 0, 1]])
    assert np.array_equal(result6, expected6), "Тест 6 не пройден"
    print('Все тесты numpy 8 пройдены')

tests_numpy_eighth()