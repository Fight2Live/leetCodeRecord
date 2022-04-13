
import time
from functools import wraps


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


def climbStairs(n: int) -> int:
    """
    爬楼梯
    假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
    每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
    """
    res = 0    # 能到达楼顶的序列[ {sum, detail:[1, 2, 1, 2, ]},  ]

    return recursionClimb(n , 1, 1)

    temps = {"sum": 0, "detail": []}
    recursionClimb(temps, n, res)
    # print(res)
    print(f'一共{res}种结果')
    return res

def recursionClimb(single, n, res, a, b):
    if n <= 1:
        return b
    return recursionClimb(n-1, b, a+b)
    step = [1, 2]
    for cur_step in step:
        if cur_step + single['sum'] < n:
            temps = {"sum": single['sum'], "detail": single['detail'].copy()}
            temps['sum'] += cur_step
            temps["detail"].append(cur_step)
            # print(cur_step, temps)
            recursionClimb(temps, n, res)
        elif cur_step + single['sum'] == n:
            # print(cur_step, single)
            # single['sum'] += cur_step
            # single["detail"].append(cur_step)
            # res.append(single)
            res += 1

class leetCode:
    def rob(self, nums: list):
        """
        你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，
        如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。给定一个代表每个房屋存放金额的非负整数数组，计算你不触动警报装置的情况
        下 ，一夜之内能够偷窃到的最高金额。

        示例 1：
        输入：[1,2,3,1]
        输出：4
        解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
        偷窃到的最高金额 = 1 + 3 = 4 。

        示例 2：
        输入：[2,7,9,3,1]
        输出：12
        解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
        偷窃到的最高金额 = 2 + 9 + 1 = 12 。
        """
        db = [[nums[0], 0],]    # db[i][]表示第i个屋子偷（0）或者不偷（1）时的最大金额
        for i in range(1, len(nums)):
            db.append([max(db[i-1][1]+nums[i], db[i-1][0]), db[i-1][0]])

        return max(db[-1])

    def maxProfit(self, prices: list) -> int:
        """
        给定一个数组 prices ，它的第i个元素prices[i] 表示一支给定股票第 i 天的价格。
        你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
        返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

        示例 1：
        输入：[7,1,5,3,6,4]
        输出：5
        解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
             注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。

        示例 2：
        输入：prices = [7,6,4,3,1]
        输出：0
        解释：在这种情况下, 没有交易完成, 所以最大利润为 0。
        """
        """
        卖出：当前手上有股票，即之前买入的，这时利润是差价
        不卖：当前手上没有股票，又或者之前买入的，没有股票时利润为0，有股票时利润为
        """

    def test(self, n:int, goods:list):
        """
        db[i] = [{'balance', 'satis', 'main'}, ]
        分买入和不买入两种状态
        但不管是否买入，都要先判断余额是否满足，主附件要求是否满足
        买入时是在之前买入的情况下买入，还是在之前不买入的情况下买入
        不买入分两种，一种是继承上一个物品的买入，一种是继承上一个物品的不买入
        :return:
        """
        db = [[{'balance':n, 'satis':0, 'main':[]}, {'balance':n, 'satis':0, 'main':[]}]]
        for i in range(len(goods)):
            good = goods[i]
            temp1 = db[i-1][0].copy()
            temp2 = db[i-1][1].copy()
            db.append([{}])

            # 买入
            after_buy = {}
            after_no_buy = {}

            ## 通关判断满意度来决定是否在之前买入的基础下
            if db[i - 1][0]['satis'] < db[i - 1][1]['satis']:
                if db[i-1][0]['balance'] <= good[0] and (good[2] == 0 or good[2] in db[i-1][0]['main']):
                    db[i][0]['balance'] = db[i - 1][0]['balance'] - good[0]
                    db[i][0]['satis'] = db[i - 1][0]['satis'] + (good[0] * good[1])
                    db[i][0]['main'] = db[i - 1][0]['main']
                    if good[2] == 0:
                        db[i][0]['main'].append(i+1)
                else:

            # 不买入
            if db[i-1][1]['balance'] <= good[0] and (good[2] == 0 or good[2] in db[i-1][1]['main']):
                if temp1['satis'] < temp2['satis']:
                    db[i].append(temp2)
                else:
                    db[i].append(temp1)
            else:
                db[i].append(temp2)

        return db[-1]

if __name__ == '__main__':
    lc = leetCode()

    t1 = [[800,2,0], [400,5,1], [300,5,1], [400,3,0],[500,2,0]]
    print(lc.test(1000, t1))
    t2 = [2,7,9,3,1]
    print(lc.rob(t2))



