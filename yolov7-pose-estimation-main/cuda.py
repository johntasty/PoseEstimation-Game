import socket
import pygame
# # Define server address and port
# SERVER_HOST = '192.168.2.73'  # Replace with Raspberry Pi IP address
# SERVER_PORT = 8000

# # Create a socket object
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# # Connect to the server
# client_socket.connect((SERVER_HOST, SERVER_PORT))


# # Send messages to the server
# while True:
   
#     message = input("Enter message to send (or type 'exit' to quit): ")
#     if message.lower() == 'exit':
#         break
#     client_socket.send(message.encode())

# # Close the connection
# client_socket.close()
image = pygame.image.load('pngs/balloonv2.png').convert_alpha()      