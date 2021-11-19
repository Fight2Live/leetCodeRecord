
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


def isPalindrome(s : str):
    """
    验证回文串
    :param s:
    :return:
    """
    temp = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    result = ''
    for i in s:
        if i in temp:
            result += i

    result = result.lower()
    print(result)
    return result == result[::-1]

def myAtoi(s: str) -> int:
    """
    atoi
    字符串转换整数
    :param s:
    :return:
    """
    temp = '0123456789'
    sym_t = '+-'
    symbol_flag = -1  # 符号位，1-正数，0-负数
    start_flag = 0   # 扫描到了数字后，开始构造整数。符号位（若存在）与首位数字相连
    result = 0
    max_num = 2 ** 31 -1
    exceed_flag = 0  # 1-超出长度

    for i in s:

        if i in sym_t and start_flag == 0:
            # 未开始构造整数时，确定符号，符号只取第一个读到的正负号
            if symbol_flag != -1:break
            if '-' == i:
                symbol_flag = 0
            else:
                symbol_flag = 1
            start_flag = 1
            continue

        if i in temp:
            start_flag = 1
            result = result * 10 + int(i)
            # 判断是否长度超过限制：
            if result > max_num:
                result = max_num
                exceed_flag = 1
                break
            continue

        elif i not in temp and start_flag == 1:
            # 开始构造整数后，遍历到非数字字符串，跳出
            break

        if i != ' ' or (i != ' ' and start_flag == 0):
            break

    if symbol_flag == 0:
        result *= -1
        if exceed_flag == 1:
            result -= 1

    return result


def strStr(haystack: str, needle: str) -> int:
    """
    strStr()
    :param haystack:
    :param needle:
    :return:
    """
    index = -1
    if needle == "":
        return 0
    needle_size = len(needle)
    for i in range(len(haystack)):
        if haystack[i : i+needle_size] == needle:
            index = i
            break
    haystack.find(needle)

    return index

def countAndSay(n: int) -> str:
    """
    外观数列
    :param n:
    :return:
    """
    result_num = init_num = '1'

    while n > 1:
        result_num = ''
        size = len(init_num)
        cur_num = init_num[0]
        cur_count = 0
        for i in range(size):
            if cur_num == init_num[i]:
                cur_count += 1
            else:
                result_num += (str(cur_count) + cur_num)
                cur_num = init_num[i]
                cur_count = 1
        result_num += (str(cur_count) + cur_num)
        init_num = result_num
        n -= 1

    return result_num

def longestCommonPrefix(strs: list) -> str:
    """
    编写一个函数来查找字符串数组中的最长公共前缀。
    如果不存在公共前缀，返回空字符串 ""。
    :param strs:
    :return:
    """
    common_str = strs[0]
    while len(common_str) > 0:
        check_flag = 1  # 0-不是公共子序列，1-是
        for s in strs[1:]:
            if s.find(common_str) != 0:
                check_flag = 0
                break
        if check_flag == 1:
            return common_str
        common_str = common_str[:-1]
    return ""


if __name__ == '__main__':

    print(longestCommonPrefix(["flower","flow","flight"]))
    print(longestCommonPrefix(["dog","racecar","car"]))




