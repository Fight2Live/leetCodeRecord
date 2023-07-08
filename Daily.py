
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        """
            # 23-7-8
            # 3. 最长子串
            https://leetcode.cn/problems/longest-substring-without-repeating-characters/
        """
        if not s: return 0
        left = 0
        lookup = set()
        n = len(s)
        max_len = 0
        cur_len = 0
        for i in range(n):
            cur_len += 1
            while s[i] in lookup:
                lookup.remove(s[left])
                left += 1
                cur_len -= 1
            if cur_len > max_len: max_len = cur_len
            lookup.add(s[i])
        return max_len


    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        """
            # 23-7-7
            # 2. 两数相和
            给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
            请你将两个数相加，并以相同形式返回一个表示和的链表。你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
            来源：力扣（LeetCode）
            链接：https://leetcode.cn/problems/add-two-numbers

        """
        data = ListNode(0)
        x1 = l1
        x2 = l2
        head = data

        while x1 or x2:
            # print(f'x1: {x1.val}, x2: {x2.val}, num = {x1.val + x2.val}')
            if not x2:
                v2 = 0
            else:
                v2 = x2.val

            if not x1:
                v1 = 0
            else:
                v1 = x1.val

            num = data.val + v1 + v2
            data.val = num % 10

            up_num = 0

            if num >= 10:
                up_num = 1

            if x1:
                x1 = x1.next
            if x2:
                x2 = x2.next

            if x1 or x2 or up_num:
                data.next = ListNode(up_num)
                data = data.next

        # print(head)
        return head

