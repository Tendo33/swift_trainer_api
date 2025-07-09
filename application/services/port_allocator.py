import socket


class PortAllocator:
    def __init__(self, port_list):
        self.port_list = port_list
    def allocate(self):
        for port in self.port_list:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("", port))
                    return port
                except OSError:
                    continue
        return None 