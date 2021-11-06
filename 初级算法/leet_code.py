
import time

"""
删除排序数组中的重复项
"""
def removeDuplicates(nums):
    n = len(nums)
    if n == 0:
        return n

    left = 0
    for right in range(n):
        if nums[right] != nums[left]:
            left += 1
            nums[left] = nums[right]


    news = nums[:left+1]
    print(f'\n原地去除数据后的数组：{nums}')
    print(f'截取后：{news}')




"""
买卖股票的最佳时机
"""
def maxProfit(prices):
    n = len(prices)
    profit_zone = []  # [i][j]表示第i天买入后，第j天卖出的利润
    for i in range(n):
        profit_zone.append([ prices[j] - prices[i] for j in range(i, n) ])
    # print(f'利润空间为：{profit_zone}')

    # 倒序遍历解空间，从后往前取出最大数相加。当取到最大数后，下一次取数的列不允许比当前取数的行大
    # buy >= sale
    max_pro = 0
    buy = sale = days = n-1
    while days >= 0:
        # print(f'正在计算第{days+1}天买入股票时的最大利润')
        cur_day_pro_zone = profit_zone[days]
        cur_max_profit = 0

        for buy in range(len(cur_day_pro_zone)):
            if cur_day_pro_zone[buy] > cur_max_profit and sale >= days + buy:
                cur_max_profit = cur_day_pro_zone[buy]
                sale = days

        if cur_max_profit > 0:
            max_pro += cur_max_profit
        days -= 1
    print(f'最大利润为：{max_pro}')


def isValidSudoku(board: list) -> bool:
    size = 9
    column_map = {}  #  每列的数字出现次数{ num1 : { col1:count, col2:count, }, }
    row_map = {}     #  每行的数字出现次数{ num1 : { row1:count, row2:count, }, }
    zone_map = {}    #  每个方块中数字出现次数{ num1 : { zone1:count, zone2:count, }, }

    # 遍历矩阵
    for row in range(size):
        cur_line = board[row]  #  当前行['1', '2', '.', ..., '.']
        row_map = {}
        for column in range(size):
            num = cur_line[column]
            if '.' == num:
                continue

            # 先定位当前所属方块：
            cur_zone = position_zone(row, column)
            print(f'board[{row}][{column}] = {num}，属于方块{cur_zone}')
            # 每列的数字出现情况
            if bool(1 - set_map_temp(num, column_map, column)):
                return False

            if bool(1 - set_map_temp(num, zone_map, cur_zone)):
                return False

            if bool(1 - set_map_temp(num, row_map, row)):
                return False

    return True



def rotate(matrix: list) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    size = len(matrix)      # 维数
    if size % 2 == 1:
        heigh = int(size / 2 + 1)
    else:
        heigh = int(size / 2)     # 最高翻转层数

    num_count = size ** 2   # 总数字
    size = size - 1         # 使用维数参与运算，-1

    for cur_heigh in range(heigh):
        cur_width = size + 1 - cur_heigh    # 当前层数的矩阵维度，即宽度
        # 维度大于3时，每一个位置翻转后的第五次都会回到起点
        if cur_width >= 3:
            rotate_count = 4
        else : rotate_count = 2
        # print(f'当前是{size + 1}维矩阵的第{cur_heigh}层，有{cur_width}维，该层的反转次数为：{rotate_count}')
        # print(f'取{cur_heigh}为起点，遍历{size - cur_heigh}次')
        for start_pos in range(cur_heigh, size - cur_heigh ):
            x = start_pos
            y = cur_heigh
            if cur_width == 3 and cur_heigh > 0: cur_width = 4
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
    # print(f'正在进行第{cur_count}次变化，当前数是{temp_data}（{x}，{y}），变化后为（{new_x}，{new_y}），被替代的值为：{inplaced_data}')
    matrix[new_x][new_y] = temp_data
    cur_count += 1
    if cur_count > rotate_count:
        return

    change_matrix(matrix, size, rotate_count, cur_count, new_x, new_y, inplaced_data)



def set_map_temp(num, temp_map, index):
    if num in temp_map:
        if index in temp_map[num]:
            temp_map[num][index] += 1
            if temp_map[num][index] == 2:
                return False
        else:
            temp_map[num][index] = 1
    else:
        temp_map[num] = {index: 1}

    return True

def position_zone(row, column):
    row, column = row+1, column+1
    if row <= 3:
        if column <= 3:return 1
        elif column <= 6: return 2
        else: return 3
    elif row <= 6:
        if column <= 3:return 4
        elif column <= 6: return 5
        else: return 6
    else:
        if column <= 3:return 7
        elif column <= 6: return 8
        else: return 9

if __name__ == '__main__':

    t = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
    t=[[1,2,3],[4,5,6],[7,8,9]]
    print(t)
    start = time.time()
    rotate(t)
    print(f'反转结果为：{t}')
    print(f'实际结果是：{[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]}')
    print(f'共花费：{time.time() - start}')



