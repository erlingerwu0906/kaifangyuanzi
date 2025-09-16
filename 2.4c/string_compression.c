#include <stdio.h>
#include <string.h>

int compress(char* chars, int charsSize) {
    if (charsSize <= 1) {
        return charsSize;
    }
    
    int write_pos = 0;
    int read_pos = 0;
    
    while (read_pos < charsSize) {
        char current_char = chars[read_pos];
        int count = 0;
        
        while (read_pos < charsSize && chars[read_pos] == current_char) {
            read_pos++;
            count++;
        }
        
        chars[write_pos++] = current_char;
        
        if (count > 1) {
            char temp[10];
            int temp_pos = 0;
            
            while (count > 0) {
                temp[temp_pos++] = (count % 10) + '0';
                count /= 10;
            }
            
            for (int i = temp_pos - 1; i >= 0; i--) {
                chars[write_pos++] = temp[i];
            }
        }
    }
    
    return write_pos;
}

int main() {
    char chars[100];
    char input[100];
    int index = 0;
    
    printf("请输入数组（格式：[\"a\",\"b\",\"b\",\"b\"]）: ");
    scanf("%*c");
    
    char c;
    int i = 0;
    while ((c = getchar()) != ']') {
        if (c == '"') {
            scanf("%c", &chars[index++]);
            scanf("%*c");
        }
    }
    
    int new_length = compress(chars, index);
    
    printf("返回 %d ，输入数组的前 %d 个字符应该是：[", new_length, new_length);
    for (int i = 0; i < new_length; i++) {
        printf("\"%c\"", chars[i]);
        if (i < new_length - 1) printf(", ");
    }
    printf("]\n");
    
    return 0;
}