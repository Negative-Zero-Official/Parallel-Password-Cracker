charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
length = int(input("Enter password length: "))
total_threads = 2048 * 256  # 524288

total_perms = len(charset) ** length
remainder = total_perms % total_threads
target_id = total_perms - remainder

suffix = []
temp = target_id
for _ in range(length):
    suffix.append(charset[temp % len(charset)])
    temp //= len(charset)

print("".join(suffix))