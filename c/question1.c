#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#define MAX_LEN 110

bool isNegative(char *num) {
    return num[0] == '-';
}

char* getNumber(char *num) {
    return isNegative(num) ? num + 1 : num;
}

int compare(char *a, char *b) {
    int len1 = strlen(a), len2 = strlen(b);
    if (len1 != len2) return len1 - len2;
    return strcmp(a, b);
}

void bigAdd(char *a, char *b, char *result) {
    bool negA = isNegative(a), negB = isNegative(b);
    char *numA = getNumber(a), *numB = getNumber(b);
    
    if (negA == negB) {
        int len1 = strlen(numA), len2 = strlen(numB);
        int maxLen = len1 > len2 ? len1 : len2;
        int sum[MAX_LEN] = {0}, index = 0, carry = 0;
        
        for (int i = 0; i < maxLen; i++) {
            int d1 = i < len1 ? numA[len1 - 1 - i] - '0' : 0;
            int d2 = i < len2 ? numB[len2 - 1 - i] - '0' : 0;
            int total = d1 + d2 + carry;
            sum[index++] = total % 10;
            carry = total / 10;
        }
        if (carry) sum[index++] = carry;
        
        if (negA) result[0] = '-';
        for (int i = 0; i < index; i++) {
            result[negA ? i + 1 : i] = sum[index - 1 - i] + '0';
        }
        result[negA ? index + 1 : index] = '\0';
        
    } else {
        int cmp = compare(numA, numB);
        char *big = cmp >= 0 ? numA : numB;
        char *small = cmp >= 0 ? numB : numA;
        bool resultNeg = (cmp >= 0) ? negA : negB;
        
        int lenBig = strlen(big), lenSmall = strlen(small);
        int diff[MAX_LEN] = {0}, index = 0, borrow = 0;
        
        for (int i = 0; i < lenBig; i++) {
            int d1 = big[lenBig - 1 - i] - '0';
            int d2 = i < lenSmall ? small[lenSmall - 1 - i] - '0' : 0;
            d1 -= borrow;
            if (d1 < d2) {
                d1 += 10;
                borrow = 1;
            } else {
                borrow = 0;
            }
            diff[index++] = d1 - d2;
        }
        
        while (index > 1 && diff[index - 1] == 0) index--;
        
        if (resultNeg) result[0] = '-';
        for (int i = 0; i < index; i++) {
            result[resultNeg ? i + 1 : i] = diff[index - 1 - i] + '0';
        }
        result[resultNeg ? index + 1 : index] = '\0';
    }
}

int main() {
    char a[MAX_LEN], b[MAX_LEN], result[MAX_LEN + 2];
    
    printf("请输入两个数：\n");
    scanf("%s %s", a, b);
    
    bigAdd(a, b, result);
    printf("结果：%s\n", result);
    
    return 0;
}