import socket

host = "127.0.0.1"
port = 8877  # GZ port

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.connect((host, port))
game_over = False

"""
message format:
[HAND_CARD]:[POSITION] [HAND_CARD]
[REQUEST_CARD]
[REQUEST_SHOW]
[NOTICE_SHOW]:[POSITION] [SHOW_CARD]
[NOTICE_CARD]:[POSITION] [CARD]
[GAME_STATUS]:[GAME_OVER|NEW_GAME]
"""

NOTICE_HAND_CARD = "HAND_CARD"
REQUEST_CARD = "REQUEST_CARD"
REQUEST_SHOW = "REQUEST_SHOW"
NOTICE_SHOW = "NOTICE_SHOW"
NOTICE_CARD = "NOTICE_CARD"
GAME_STATUS = "GAME_STATUS"
GAME_OVER = "GAME_OVER"

MESSAGE_MAX_LENGTH = 1024


def main():
    while not game_over:
        data = socket.recv(MESSAGE_MAX_LENGTH).decode().split(":")
        message_type = data[0]
        if message_type == '':
            continue
        elif message_type == REQUEST_SHOW:
            pass
        elif message_type == REQUEST_CARD:
            pass
        elif message_type == NOTICE_SHOW:
            position, card = data[1].split(" ")
            print(position)
            print(card)
            pass
        elif message_type == NOTICE_CARD:
            position, card = data[1].split(" ")
            print(position)
            print(card)
            pass
        elif data == GAME_STATUS:
            pass


if __name__ == '__main__':
    main()
