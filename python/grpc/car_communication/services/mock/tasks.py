from dataclasses import dataclass

from services.task_manager import TaskManager, Product

@dataclass
class MockTask:
    nearest_router_id: str
    product_id: int
    product_name: str
    product_qr_code_metadata: str


__car_id = 'A-001'
__MOCK_TASKS = [
    MockTask(
        nearest_router_id = '5-1',
        product_id = 1,
        product_name = 'TargetBox',
        product_qr_code_metadata = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    ),
    MockTask(
        nearest_router_id = '7',
        product_id = 2,
        product_name = 'TargetBox2',
        product_qr_code_metadata = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    ),
]


def init_mock_tasks(car_id: str = 'A-001') -> None:
    global __car_id
    __car_id = car_id

def add_mock_task():
    if len(__MOCK_TASKS) == 0: 
        print('No more mock task, failed add to task manager')
        TaskManager.remove_active_task(__car_id)
        return
    mock_task = __MOCK_TASKS.pop(0)
    product = Product(
        id = mock_task.product_id,
        name = mock_task.product_name,
        qr_code_metadata = mock_task.product_qr_code_metadata,
    )
    TaskManager.create_active_task(
        car_id = __car_id,
        product = product,
        target_router_id = mock_task.nearest_router_id,
    )

def _test():
    car_id = 'A-002'
    init_mock_tasks(car_id = car_id)
    add_mock_task()
    active_task = TaskManager.get_active_task(car_id=car_id)
    print(f'CarID: {car_id}')
    print(active_task)
    add_mock_task()
    add_mock_task()
    active_task = TaskManager.get_active_task(car_id=car_id)
    print(active_task)

if __name__ == '__main__':
    _test()
