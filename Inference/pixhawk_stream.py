import asyncio
from mavsdk import System

async def main():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyACM0:115200")
    print("Waiting for drone to connect...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to Pixhawk.")
            break

    async for position in drone.telemetry.position_velocity_ned():
        print(f"x={position.position.north_m:.2f}, y={position.position.east_m:.2f}, z={position.position.down_m:.2f}")
    asyncio.run(main())