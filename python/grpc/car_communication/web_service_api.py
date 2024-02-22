from dataclasses import dataclass


class RouterID(str):
    pass

@dataclass
class Product:
    id: int
    name: str
    qr_code_metadata: str

@dataclass
class WebServiceResponse:
    product: Product | None
    route: list[RouterID]

@dataclass
class WebServiceRequest:
    car_id: str
    nearest_router_id: str

@dataclass
class Task:
    product: Product
    target_router: RouterID


class TaskManager:
    _active_tasks: dict[str, Task] = {}

    @classmethod
    def create_active_task(
        cls, 
        car_id:str, 
        product: Product, 
        target_router: RouterID,
    ) -> None:
        task = Task(product=product,target_router=target_router)
        cls._active_tasks.update({car_id:task})
    
    @classmethod
    def get_active_task(cls, car_id:str) -> Task | None:
        task = cls._active_tasks.get(car_id)
        return task

    @classmethod
    def remove_active_task(cls, car_id:str) -> None:
        cls._active_tasks.pop(car_id)

class RouteManager:
    pass

class WebService:
    _no_active_task_response = WebServiceResponse(None,[])

    def send_request(self, request: WebServiceRequest) -> WebServiceResponse:
        task = TaskManager.get_active_task(request.car_id)
        if not(task): return self._no_active_task_response
        route = [RouterID(i) for i in '1234'] #TODO
        response = WebServiceResponse(
            product = task.product,
            route = route,
        )
        return response


def _test():
    web_service = WebService() 
    # init demo car
    car_id = 'A-001'
    nearest_router_id = '9'
    # init demo car task
    product_nearest_router = '5'
    product = Product(1,'Box of apples', 'qr_box_of_apples')
    TaskManager.create_active_task(
        car_id=car_id,
        product=product,
        target_router=RouterID(product_nearest_router),
    )
    # get response from web service
    request = WebServiceRequest(
        car_id=car_id, 
        nearest_router_id=nearest_router_id,
    )
    response = web_service.send_request(request)
    print(response)


if __name__ == '__main__':
    _test()
