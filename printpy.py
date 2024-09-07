print('Enter your name:')
user_thresh = input()
print(user_thresh)
with open("Python.txt", "w+") as f:
    f.write(user_thresh)
