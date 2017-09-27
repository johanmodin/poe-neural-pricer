# Code taken from https://stackoverflow.com/questions/23602412/only-download-a-part-of-the-document-using-python-requests
# Credit to https://stackoverflow.com/users/1005215/nehal-j-wani

# Downloads the first 512 bytes to quickly get next_change_id without
# having to download the full page which takes ~7 seconds for me.
# This may be used to multithread the dataretriever
import socket
import time
def get_next_id(id):
    MESSAGE = \
    "GET /public-stash-tabs?id=" + id + " HTTP/1.1\r\n"  \
    "HOST: api.pathofexile.com\r\n" \
    "User-Agent: Custom/0.0.1\r\n" \
    "Accept: */*\r\n\n"

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('api.pathofexile.com', 80))
    s.send(MESSAGE.encode('utf-8'))

    time.sleep(0.1)

    curr_size = 0
    data = ""
    while curr_size < 512:
        data += s.recv(512 - curr_size).decode('utf-8')
        curr_size = len(data)

    s.close()

    return data.split('","stashes"')[0].split('{"next_change_id":"')[1]
