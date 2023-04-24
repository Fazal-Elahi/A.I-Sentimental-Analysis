while True:
    # choice = input("Please enter a number between 1-5:\n1. Happiness\n2. Sadness\n3. Anger\n4. Surprise\n5. Fear\nEmotion: ")
    if choice.isdigit():
        choice = int(choice)
        if 1 <= choice <= 5:
            break
    print("Invalid input. Please enter a number between 1 and 5.")

emotions = {1: "Happiness",2: "Sadness",3: "Anger",4: "Surprise",5: "Fear"}
print(f"Simulating {emotions[choice]}")
