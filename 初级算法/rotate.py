
def rotate(matrix: list) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    size = len(matrix)      # 维数
    if size % 2 == 1:
        heigh = int(size / 2 + 1)
    else:
        heigh = int(size / 2)     # 最高翻转层数

    if size <= -9:
        change_matrix(matrix, 1, 4, 1, 0, 0, matrix[0][0])
    else :
        num_count = size ** 2   # 总数字
        size = size - 1         # 使用维数参与运算，-1

        for cur_heigh in range(heigh):
            cur_width = size + 1 - cur_heigh    # 当前层数的矩阵维度，即宽度
            # 维度大于3时，每一个位置翻转后的第五次都会回到起点
            if cur_width >= 3:
                rotate_count = 4
            else : rotate_count = 4
            print(f'当前是{size + 1}维矩阵的第{cur_heigh}层，有{cur_width}维，该层的反转次数为：{rotate_count}')
            print(f'取{cur_heigh}为起点，遍历{size - cur_heigh}次')
            for start_pos in range(cur_heigh, size - cur_heigh ):
                x = start_pos
                y = cur_heigh
                if cur_width == 3 and cur_heigh > 0: cur_width = size + 1
                change_matrix(matrix, cur_width - 1, rotate_count, 1, x, y, matrix[x][y])



def change_matrix(matrix: list, size, rotate_count, cur_count, x, y, temp_data) :
    """
    顺时针旋转数组
    :param matrix:          初始数组
    :param size:            宽度
    :param rotate_count:     数字总个数
    :param cur_count:       当前以变动次数
    :param x:               当前需要移动的元素的X
    :param y:               当前需要移动的元素的Y
    :param temp_data:       上一次翻转后目标位置的数
    :return:
    """
    new_x = y
    new_y = size - x
    inplaced_data = matrix[new_x][new_y]
    print(f'正在进行第{cur_count}次变化，当前数是{temp_data}（{x}，{y}），变化后为（{new_x}，{new_y}），被替代的值为：{inplaced_data}')
    matrix[new_x][new_y] = temp_data
    cur_count += 1
    if cur_count > rotate_count:
        return

    change_matrix(matrix, size, rotate_count, cur_count, new_x, new_y, inplaced_data)

t = [[1,2,3],[4,5,6],[7,8,9]]
t_hat =[[7,4,1],[8,5,2],[9,6,3]]

t1 = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
t1_hat = [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]

t3 = [[1,2],[3,4]]
t3_hat = [[3,1],[4,2]]

t4 = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]
t4_hat = [[21,16,11,6,1],[22,17,12,7,2],[23,18,13,8,3],[24,19,14,9,4],[25,20,15,10,5]]

rotate(t4)
print(f'反转结果为：{t4}')
print(f'实际结果是：{t4_hat}')




