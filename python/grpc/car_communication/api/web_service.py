from services.route_manager import RouteManager
from services.task_manager import (
    TaskManager,
    Product,
)
from dataclasses import dataclass


@dataclass
class WebServiceResponse:
    product: Product | None
    route: list[str]

@dataclass
class WebServiceRequest:
    car_id: str
    nearest_router_id: str


class WebService:
    _no_active_task_response = WebServiceResponse(None,[])

    def send_request(self, request: WebServiceRequest) -> WebServiceResponse:
        task = TaskManager.get_active_task(request.car_id)
        if not(task): return self._no_active_task_response
        route = RouteManager.get_shortest_route(
            nearest_router_id=request.nearest_router_id,
            target_router_id=task.target_router_id,
        )
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
    product = Product(
        id=1,
        name='Box of apples',
        qr_code_metadata='qr_box_of_apples',
    )
    product_nearest_router = '5'
    TaskManager.create_active_task(
        car_id=car_id,
        product=product,
        target_router_id=product_nearest_router,
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
