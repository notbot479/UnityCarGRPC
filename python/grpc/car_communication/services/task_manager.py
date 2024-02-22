from dataclasses import dataclass


@dataclass
class Product:
    id: int
    name: str
    qr_code_metadata: str

@dataclass
class Task:
    product: Product
    target_router_id: str


class TaskManager:
    _active_tasks: dict[str, Task] = {}

    @classmethod
    def create_active_task(
        cls, 
        car_id:str, 
        product: Product, 
        target_router_id: str,
    ) -> None:
        task = Task(product=product,target_router_id=target_router_id)
        cls._active_tasks.update({car_id:task})
    
    @classmethod
    def get_active_task(cls, car_id:str) -> Task | None:
        task = cls._active_tasks.get(car_id)
        return task

    @classmethod
    def remove_active_task(cls, car_id:str) -> None:
        cls._active_tasks.pop(car_id)


def _test():
    car_id = 'A-001'
    product_nearest_router_id = '5'
    product1 = Product(1,'BoxOfOranges', 'qr_box_of_oranges')
    product2 = Product(2,'Milk', 'qr_milk_box')

    # add task for car
    TaskManager.create_active_task(
        car_id=car_id,
        product=product1,
        target_router_id=product_nearest_router_id,
    )
    print(TaskManager.get_active_task(car_id))
    
    # update task for car
    TaskManager.create_active_task(
        car_id=car_id,
        product=product2,
        target_router_id=product_nearest_router_id,
    )
    print(TaskManager.get_active_task(car_id))

    # delete task
    TaskManager.remove_active_task(car_id)
    print(TaskManager.get_active_task(car_id))


if __name__ == '__main__':
    _test()
