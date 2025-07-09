from .port_allocator import PortAllocator


class DeployHandler:
    def __init__(self, port_list=None):
        # 端口池可通过配置注入，这里用硬编码示例
        self.port_list = port_list or [8001, 8002, 8003]

    def handle(self, deploy_params):
        allocator = PortAllocator(self.port_list)
        port = allocator.allocate()
        if not port:
            raise Exception("无可用端口，请稍后重试")
        deploy_params.port = port
        # 这里可扩展实际部署逻辑，如启动服务、写入状态等
        return port 