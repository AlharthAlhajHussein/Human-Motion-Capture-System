import asyncio
import websockets
import json
import threading
from collections import deque

class KeypointServer:
    """
    A WebSocket server that broadcasts keypoint data to all connected clients.
    It runs in a separate thread to avoid blocking the main application.
    """
    def __init__(self, port):
        self.port = port
        self.host = '0.0.0.0'     #'localhost'
        self.clients = set()
        self.server_thread = threading.Thread(target=self._start_server)
        self.loop = None
        self.server = None  # To hold the server object
        self.message_queue = deque()

    async def _handler(self, websocket):
        """
        Handles new WebSocket connections. Adds the client to the set
        of connected clients and keeps the connection alive.
        """
        print(f"Unity client connected from {websocket.remote_address}")
        self.clients.add(websocket)
        try:
            # Keep the connection open and listen for any potential messages
            # or disconnection events from the client.
            async for message in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            print("Unity client disconnected.")
        finally:
            self.clients.remove(websocket)

    async def _broadcast_loop(self):
        """
        Continuously checks the queue for new messages and broadcasts them.
        """
        try:
            while True:
                if self.message_queue:
                    message = self.message_queue.popleft()
                    if self.clients:
                        # Use asyncio.gather to send messages to all clients concurrently.
                        await asyncio.gather(
                            *[client.send(message) for client in self.clients]
                        )
                await asyncio.sleep(0.001)  # Sleep briefly to yield control
        except asyncio.CancelledError:
            print("Broadcast loop cancelled.")

    async def _main(self):
        """The main async method to run the server and broadcast loop."""
        self.server = await websockets.serve(self._handler, self.host, self.port)
        print(f"WebSocket server is running on ws://{self.host}:{self.port}")
        broadcast_task = asyncio.create_task(self._broadcast_loop())
        try:
            await self.server.wait_closed()
        finally:
            broadcast_task.cancel()

    def _start_server(self):
        """
        The entry point for the server thread. Sets up and runs the
        asyncio event loop.
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main())
        except Exception as e:
            print(f"Error in server's main loop: {e}")
        finally:
            self.loop.close()
            print("Server loop closed.")

    def start(self):
        """Starts the server thread."""
        self.server_thread.start()
        # The print statement is moved to _main to ensure it prints when the server is actually running.

    def stop(self):
        """Stops the server and the event loop."""
        if self.server and self.loop:
            self.loop.call_soon_threadsafe(self.server.close)
        print("WebSocket server stopping...")
        self.server_thread.join(timeout=2) # Wait for the thread to finish

    def broadcast(self, data):
        """
        Adds keypoint data to the message queue to be broadcast.
        This method is called from the main processing thread.
        """
        try:
            message = json.dumps(data)
            self.message_queue.append(message)
        except Exception as e:
            print(f"Error serializing data for broadcast: {e}") 