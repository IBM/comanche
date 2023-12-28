import os

# Function to create a file of the specified size in bytes
def create_file(file_size_bytes, file_name):
    with open(file_name, 'wb') as f:
        f.write(b'\0' * file_size_bytes)

# Define the file sizes in bytes (from 5MB to 10MB)
file_sizes = [5 * 1024 * 1024, 6 * 1024 * 1024, 7 * 1024 * 1024, 8 * 1024 * 1024, 9 * 1024 * 1024, 10 * 1024 * 1024]

# Create files of different sizes
for size in file_sizes:
    file_name = f"{size // (1024 * 1024)}MB_file.txt"
    create_file(size, file_name)
    print(f"Created {file_name} ({size // (1024 * 1024)}MB)")

print("Files created successfully.")

