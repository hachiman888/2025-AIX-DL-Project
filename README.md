# D* 알고리즘에 대한 이해 및 분석

## **구성원**

* 장봉호옥

  * 학과: 스포츠과학

  * email: 15737390567@163.com

* 증준상
  * 학과: 정보시스템학과
  * email: goblin8456@gmail.com

## **1. 개요**           

### **주제 선택의 이유**



​         수업 중 교수님이 제시한 8-puzzle 문제는 매우 흥미로웠으며, 이 문제를 해결하기 위해 A* 알고리즘을 활용한 간단한 경로 계획이 가능합니다. 그러나 우리는 ***정적 환경***에서는 A* 알고리즘이 경로 계획 문제를 잘 해결할 수 있지만, ***동적 환경***에서는 다소 부족함이 있음을 알 수 있습니다. 이 글에서는 A* 알고리즘의 한계점과 이러한 한계를 보완하기 위해 제안된 D\*(Dynamic A\*) 알고리즘의 **역사**, **작동 원리**, **적용 사례**, **장단점** 및 **코드 예시**를 소개하고자 합니다.





## 2. **D\*알고리즘이 왜 필요합니까?**

### 2.1 **A\*알고리즘은 무엇입니까?**

 **A\*알고리즘 정의**:

A* 알고리즘(A* search algorithm)은 그래프 평면에 여러 노드가 있는 경로에서 최저 통과 비용을 산출하는 알고리즘입니다. 주로 게임 내 NPC의 이동 계산이나 온라인 게임 봇의 이동 계산에 사용됩니다.

A* 알고리즘은 출발 꼭짓점으로부터 목표 꼭짓점까지의 최적 경로를 탐색하기 위한 것이다. 이를 위해서는 각각의 꼭짓점에 대한 평가 함수를 정의해야 한다. 이를 위한 평가 함수 `f(n)`은 다음과 같다.
$$
f(n) = g(n) + h(n)
$$

* `g(n)` :출발 꼭짓점으로부터 꼭짓점 n까지의 경로 가중치
* `h(n)` :꼭짓점 n으로부터 목표 꼭짓점까지의 추정 경로 가중치



**A\* 알고리즘 범례**:

![image](https://github.com/hachiman888/2025-AIX-DL-Project/blob/main/img/Astar_progress_animation.gif)

![image](https://github.com/hachiman888/2025-AIX-DL-Project/blob/main/img/Astarpathfinding.gif)

이 알고리즘은 ***최상우선탐색(Best-First Search)***과 다익스트라 ***알고리즘(Dijkstra's algorithm)***의 장점을 결합하였습니다: 휴리스틱 탐색을 통해 알고리즘 효율을 높일 동시에 (평가 함수가 단조성을 만족할 경우) 최적 경로를 찾을 수 있음을 보장합니다.





### 2.2 **A\*알고리즘의 제한점**

1. **메모리 소비량이 많음**

   이것은 A* 알고리즘의 가장 두드러진 단점입니다.

   * **원인**: A* 알고리즘은 탐색 대기 중인 노드와 이미 탐색한 노드를 저장하기 위한 **열린 목록**과 **닫힌 목록**을 유지해야 합니다. 검색 공간이 거대할 경우(예: 매우 촘촘한 그리드 또는 복잡한 그래프) 저장해야 하는 노드 수가 기하급수적으로 증가하여大量의 메모리를 소비합니다.

2. **휴리스틱 함수에 대한 의존성**

   - **부정확한 휴리스틱 함수**: `h(n)`이 실제 비용을 과소평가(허용 가능)하면 A*는 최적 경로를 찾는 것을 보장하지만, 너무 심하게 과소평가되면 다익스트라 알고리즘처럼 동작하여 효율성이 떨어지고 불필요한 노드를大量 탐색하게 됩니다.
   - **허용 불가능한 휴리스틱 함수**: `h(n)`이 실제 비용을 과대평가(허용 불가능)하면 A*는 **최적 경로를 보장할 수 없지만**, 더 빠르게 실행 가능한 경로를 찾을 수는 있습니다.

3. **3차원 이상의 공간에서 성능 저하**

   * A* 알고리즘은 2차원 평면 맵에서 뛰어나지만, 차원이 증가함에 따라 탐색해야 하는 노드 수가 급격히 증가하는 **차원의 저주** 문제에 직면합니다.

4. **동적 환경 처리 불가**

   - **원인**: 경로를 계산하는 동안 맵이 고정되어 있다고 가정합니다. 계산이 완료된 후 환경에 새로운 장애물이 나거나 기존 장애물이 이동하면 A*는 실시간으로 효과적으로 대응할 수 없습니다.



### 2.3 **D\*알고리즘의 역사**

기존 A* 알고리즘은 정적 지도, 즉 경로 계획 과정에서 지도 내 장애물이 변하지 않는다는 가정 하에 사용됩니다. 그러나 실제 상황에서는 로봇이 갑자기 장애물이 나타나거나 사라지는 상황을 마주칠 수 있으며, 지도도 변화합니다. *<u>**D\* 알고리즘은 바로 이러한 동적 환경에서의 경로 계획 문제를 해결합니다.**</u>*

**D\* 알고리즘 작동 범례**:

![image](https://github.com/hachiman888/2025-AIX-DL-Project/blob/main/img/b585763e54494849b51929fa51ef5342.gif)


D* 알고리즘은 Anthony Stentz에 의해 1994년에 처음 제안되었으며, 동적 환경에서의 경로 계획 문제에 대처하는 것이 목적입니다. 이 알고리즘의 기원은 **자율 주행**, **이동 로봇**, 특히 **실시간으로 지도 변화를 처리해야 하는 분야와 밀접한 관련이 있습니다**. D* 알고리즘은 점진적 경로 계획 알고리즘으로, 환경이 변화할 때 이전에 계산된 경로를 바탕으로 처음부터 전체 경로를 재계산하지 않고 점진적으로 업데이트하여 계산 자원을 절약하고 실시간 대응 능력을 향상시킵니다.



## 3. **D\* 알고리즘의 핵심 개념과 작동 원리**

### **3.1 핵심 개념**: **역방향 탐색과 비용 전파**

* **역방향 탐색**:

   A* 알고리즘이 시작점에서 목표점으로 향하는 방식과 달리, D* 알고리즘의 초기 경로 계획은 목표점에서 시작하여 역방향으로 시작점을 탐색합니다. 이는 알고리즘이 처음부터 지도 상의 모든 지점에서 목표점까지의 최적 비용을 계산해 놓음을 의미합니다. 따라서 로봇이 시작점에 있을 때, 이 미리 계산된 기울기를 따라 "내려가기"만 하면 목표지점에 도달할 수 있습니다.

* **비용 전파 (재계획의 핵심)**:

  로봇이 이동 중 특정 간선(예: 노드 A에서 노드 B로의 경로)의 비용 증가를 감지했을 때(예: 새로운 장애물 발견), D* 알고리즘은 다음과 같이 효율적으로 대응합니다:

  * 변화 지점 식별: 노드 A와 B와 같이 영향을 받은 노드들을 '변화됨'으로 표시합니다.
  * 전파 시작: 마치 돌을 물에 던져 파문이 퍼지듯, 알고리즘은 변화 지점을 중심으로 비용 증가 정보를 주변(특히 로봇의 현재 위치 방향을 중심으로)으로 전파합니다.

  * 국부 수정: 이 변화의 영향을 받은 노드들의 비용과 경로만을 재계산하여, 최종적으로 현재 위치에서 목표지점까지의 새로운 유효 경로를 신속하게 도출합니다.



### **3.2 작동 원리 **:

#### A* 알고리즘 기초 복습

먼저, A* 알고리즘이 경로를 계산하는 기본 공식은 다음과 같습니다:
$$
f(n) = g(n) + h(n)
$$

* `f(n)` :현재 노드의 평가 함수는 시작점부터 목표지점까지의 예상 총 비용을 나타냅니다.

* `g(n)` :출발 꼭짓점으로부터 꼭짓점 n까지의 경로 가중치
* `h(n)` :꼭짓점 n으로부터 목표 꼭짓점까지의 추정 경로 가중치



#### D* 알고리즘 기초 단계

D* 알고리즘은 이를 기반으로 개선을 이루었으며, 변화하는 환경에 대응하기 위해 여러 새로운 공식을 도입했습니다.



1) 초기화: 목표 노드 값 계산

   A* 알고리즘과 유사하게, D* 알고리즘은 시작 시 각 노드를 초기화하고 초기 경로 비용을 계산합니다. 각 노드에 대해 D* 알고리즘은 휴리스틱 값 h(n)을 계산하며, 두 가지 중요한 값을 정의합니다:

   * `g(n)`: 시작점에서 현재 노드까지의 실제 비용

   * `rhs(n)`: D\*에서 rhs는 갱신된 비용(D\*에만 있는 독특한 개념)입니다. 이는 현재 노드에서 목표 노드까지의 비용을 나타내며, 환경 변화 시 동적으로 갱신됩니다.

     목표 노드의 초기값은 다음과 같이 설정됩니다:

     
     $$
     rhs(goal) = 0,g(goal) = \infty
     $$

   

2) 갱신 규칙: 국부 갱신

   D* 알고리즘의 핵심 개념은 장애물이 변화했을 때 전체 경로를 처음부터 재계산하지 않고, 영향 받은 지역에 기반하여 경로를 국부적으로 갱신한다는 점입니다.

   

   매번 노드 갱신 시 D*는 다음과 같은 두 가지 갱신 규칙을 사용하여 새로운 값을 계산합니다:

   `g(n)` 값 갱신:

   
   $$
   g(n)=\min_{s \in S_n}[c(n,s)+g(s)]
   $$
   

   `S(n)`:현재 노드 n의 이웃 노드 집합

   `c(n,s)`:노드 n에서 이웃 노드 s까지의 이동 비용(일반적으로 1 또는 유클리드 거리)

   `g(s)`:이웃 노드 s에서 시작점까지의 실제 비용

   `rhs(n)` 갱신:

   
   $$
   rhs(n) = \min_{s\in S_n}[c(n,s)+g(s)]
   $$
   

   이 두 공식은 노드의 최소 비용을 계산하며, 점진적으로 경로의 추정값을 갱신합니다.

3) 우선순위 큐 갱신

   D* 알고리즘은 갱신 대기 중인 노드들을 관리하기 위해 하나의 우선순위 큐를 사용합니다. 큐에 있는 노드들은 `f(n) = g(n) + h(n)` 순서로 정렬되어 순차적으로 갱신됩니다. 우선순위 큐의 이러한 갱신 방식은 경로 계산의 효율성을 보장합니다.

4) 역방향 탐색

   D* 알고리즘의 독특한 점은 시작점이 아닌 목표 노드에서부터 역방향으로 경로를 탐색한다는 것입니다. 이러한 역방향 탐색 방식은 지도 변화 발생 시 경로를 국부적으로만 조정할 수 있게 해줍니다.

   

   역방향 갱신 과정에서 D* 알고리즘은 갱신된 값을 기반으로 새로운 최단 경로를 계산합니다. 장애물 변화가 발생하면 영향을 받은 영역을 확인하고 경로 추정값을 다음과 같이 갱신합니다:

   
   $$
   g(n)=\min_{s \in S_n}[c(n,s)+g(s)]
   $$
   

   이 갱신 과정은 목표점에서 시작점으로 영향 범위를 점차 전파하며 단계적으로 경로를 조정합니다.

5) 점진적 갱신

   D* 알고리즘의 점진적 갱신 개념은 지역적 경로 조정에 기반하여 전체 경로를 재계산하는 계산량을 줄입니다. 장애물이 이동할 때 D* 알고리즘은 전역 경로를 재계산하지 않고 직접 영향을 받은 경로 부분만 재계산합니다.

6) 최종 경로 계산

   경로 계산이 완료되면 D* 알고리즘은 시작점에서 목표지점까지의 최단 경로를 출력합니다. 갱신 과정에서 장애물이 사라지거나 다른 변화가 발생하면 D* 알고리즘은 실시간으로 경로를 조정하여 로봇이 항상 최적의 경로를 따라 전진하도록 보장합니다.

   

   D* 알고리즘의 핵심은 A*의 휴리스틱 탐색 사상을 기반으로 점진적 갱신과 역방향 탐색 기술을 결합하여 효율적인 동적 경로 계획을 구현하는 데 있습니다. 다음 요소들을 통해 구현됩니다:

   

   `g(n)`: 시작점에서 현재 노드까지의 실제 비용을 나타냅니다.
   `rhs(n)`: 현재 노드에서 목표 노드까지의 비용을 나타내며 환경 변화 시 동적으로 갱신됩니다.
   우선순위 큐: 최소 비용 경로의 신속한 선택과 갱신을 돕습니다.

   

   이러한 공식과 갱신 규칙들은 D* 알고리즘이 동적 환경에서의 경로 계획 문제를 효율적으로 처리할 수 있게 하며, 계산량을 줄이고 실시간 대응 능력을 향상시킵니다.



## **4.D\*알고리즘 가족및 적용 사례**

### **4.1 D\*알고리즘의 세 가지 버전**

| 알고리즘 특성      |                D* (origin)                 |                          Focused D*                          |                           D* Lite                            |
| :----------------- | :----------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **핵심 겨념**      | 동적 환경에서의 역방향 탐색 및 점진적 갱신 | D* 알고리즘 기반에 **휴리스틱**을 도입하여 "유망한" 노드를 우선 처리합니다 | **논리가 더욱 명확하고 간결해져**, **시작점에서 정방향 탐색**을 수행하면서 **우선순위 키**를巧妙하게 설정하여 역방향 효과를 모방합니다 |
| **주요 개선점**    |                    기초                    |                  **재계획 탐색 효율 향상**                   | **성능이 동등하거나 더 우수하며, 구현과 이해가 더 쉽습니다** |
| **권장 사용 사례** |               기초 개념 이해               |               대규모 환경에서의 신속한 재계획                |        **현대 동적 경로 계획 프로젝트의 최상의 선택**        |



### **4.2 생확 속에 적용 사례**

- 로봇 내비게이션: 특히 미지 환경 탐색이나 동적으로 변화하는 환경 상황에서 사용됨
- 자율주행: 도로상의 갑자기 나타나는 장애물에 대응
- 게임 AI: 동적으로 변화하는 게임 세계에서 게임 캐릭터의 지능적인 경로 탐색 구현
- 드론 검측: 복잡한 환경에서 갑자기 나타나는 장애물 회피



##  **5.D\* 알고리즘의 장단점**

**D\* 알고리즘 장단점 비교**

| 장점                                                    | 단점                                                        |
| :------------------------------------------------------ | :---------------------------------------------------------- |
| 점진적 경로 갱신으로 계산 시간과 자원 절약              | 구현이 복잡하여 A* 알고리즘보다 구현이 어려움               |
| 동적 환경 적응 가능, 장애물 변화 처리 능력              | 대규모 환경 변화 시 성능 저하, 계산 효율성 감소             |
| 역방향 탐색으로 시작점부터의 경로 재계산 방지           | 지역 최적 경로 문제 발생, 최적 경로를 반환하지 않을 수 있음 |
| 실시간 성능 우수, 실시간 시스템 및 동적 응용场景에 적합 | 메모리 소비량 많음, 더 많은 정보 저장 필요                  |
| 계산 효율성 높음, 특히 장애물 변화가 적은 경우          | 밀집 장애물 환경에서 효과 미흡                              |
| 영향 받은 영역의 경로만 갱신하여 계산 자원 절약         | 성능 병목 현상, 복잡한 환경에서 성능 문제 발생 가능         |



## **6.코드 예시**(python)

### **6.1필요 환경**

```python
import math
from sys import maxsize
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
import numpy as np
```



### **6.2상태 설치**

```python
# 지도 상의 각 셀 상태를 나타냄
class State(object):
    def __init__(self, x, y):
        self.x = x  # 행 좌표
        self.y = y  # 열 좌표
        self.parent = None  # 경로 추적을 위한 이전 노드
        self.state = "."  # 상태 기호: . 빈칸, # 장애물, s 시작점, e 종점, * 경로
        self.t = "new"  # 상태 표시: new/open/close
        self.h = 0  # 휴리스틱 비용 값
        self.k = 0  # 최소 경로 추정 함수
 
    def cost(self, state):
        # 장애물이 있을 경우 최대값 반환, 그렇지 않으면 유클리드 거리 계산
        if self.state == "#" or state.state == "#":
            return maxsize
        return math.hypot(self.x - state.x, self.y - state.y)
 
    def set_state(self, state):
        # 셀 상태 설정
        if state not in ["s", ".", "#", "e", "*"]:
            return
        self.state = state
```



### **6.3지도 설치**

```python
# 전체 지도를 나타내며 그리드와 장애물을 포함
class Map(object):
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.map = self.init_map()
 
    def init_map(self):
        # State 객체의 2차원 리스트로 지도 초기화
        return [[State(i, j) for j in range(self.col)] for i in range(self.row)]
 
    def get_neighbers(self, state):
        # 한 셀 주변 8방향의 이웃 획득 (대각선 포함)
        state_list = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                nx, ny = state.x + i, state.y + j
                if 0 <= nx < self.row and 0 <= ny < self.col:
                    state_list.append(self.map[nx][ny])
        return state_list
 
    def set_obstacle(self, point_list):
        # 지도상에 장애물 설정
        for x, y in point_list:
            if 0 <= x < self.row and 0 <= y < self.col:
                self.map[x][y].set_state("#")
```



### **6.4 알고리즘**

```python
# D* 경로 계획 알고리즘 구현
class Dstar(object):
    def __init__(self, maps):
        self.map = maps
        self.open_list = set()  # 처리 대기 중인 노드 저장
        self.frames = []  # 각 프레임 애니메이션 상태 저장
        self.start = None
        self.end = None
 
    def process_state(self):
        # 핵심 상태 처리 함수, D*의 세 가지 경우에 따라 경로 정보 갱신
        x = self.min_state()  # open list에서 비용이 가장 작은 노드 획득
        if x is None:
            return -1  # 노드가 없으면 -1 반환
        k_old = self.get_kmin()  # 현재 최소 비용 k 값 획득
        self.remove(x)  # 노드 x를 open list에서 제거
        if k_old < x.h:  # 현재 k가 x의 휴리스틱 비용보다 작은 경우
            # 이웃 노드 처리, 경로와 비용 갱신
            for y in self.map.get_neighbers(x):
                if y.h <= k_old and x.h > y.h + x.cost(y):
                    x.parent = y
                    x.h = y.h + x.cost(y)
        elif k_old == x.h:  # k 값이 같은 경우, 노드 갱신
            for y in self.map.get_neighbers(x):
                if y.t == "new" or (y.parent == x and y.h != x.h + x.cost(y)) or (y.parent != x and y.h > x.h + x.cost(y)):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
        else:  # k 값이 더 큰 경우, 이웃 노드 갱신
            for y in self.map.get_neighbers(x):
                if y.t == "new" or (y.parent == x and y.h != x.h + x.cost(y)):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
                elif y.parent != x and y.h > x.h + x.cost(y):
                    self.insert(y, x.h)
                elif y.parent != x and x.h > y.h + x.cost(y) and y.t == "close" and y.h > k_old:
                    self.insert(y, y.h)
        return self.get_kmin()  # 최소 k 값 반환
 
    def min_state(self):
        # open list에서 k가 가장 작은 노드 획득
        return min(self.open_list, key=lambda x: x.k) if self.open_list else None
 
    def get_kmin(self):
        # open list에서 가장 작은 k 값 획득
        return min((x.k for x in self.open_list), default=-1)
 
    def insert(self, state, h_new):
        # 노드를 open list에 삽입하고 상태 설정
        if state.t == "new":
            state.k = h_new  # 새 노드인 경우 k 값 직접 설정
        elif state.t == "open":
            state.k = min(state.k, h_new)  # 노드가 이미 open list에 있는 경우 더 작은 k 값 선택
        elif state.t == "close":
            state.k = min(state.h, h_new)  # 노드가 이미 닫힌 경우 더 작은 비용 선택
        state.h = h_new
        state.t = "open"  # open 상태로 설정
        self.open_list.add(state)  # 노드를 open list에 추가
 
    def remove(self, state):
        # 노드를 open list에서 제거
        if state.t == "open":
            state.t = "close"  # close 상태로 설정
        self.open_list.remove(state)
 
    def modify_cost(self, x):
        # 노드 비용 수정, open list 재삽입 트리거
        if x.t == "close":
            self.insert(x, x.parent.h + x.cost(x.parent))  # 재삽입 및 비용 갱신
 
    def run(self, start, end):
        path_length = 0  # 경로 길이 통계 초기화
        path_cost = 0    # 경로 이동 비용 초기화
        self.start = start
        self.end = end
        self.open_list.add(end)  # 종점에서 시작점으로 역추적
        while True:
            self.process_state()  # 현재 상태 처리
            if start.t == "close":  # 시작점이 닫혔으면 경로 찾음
                break
        start.set_state("s")  # 시작점 상태 설정
        s = start
        # 경로 역추적 및 상태 설정
        while s != end:
            s.set_state("s")
            path_length += 1  # 각 단계 경로 1 증가
            path_cost += s.cost(s.parent) if s.parent else 0  # 비용 누적
            self.capture_frame()  # 현재 프레임 캡처
            s = s.parent
        s.set_state("e")  # 종점 상태 설정
        self.capture_frame()
 
        # 새 장애물 시뮬레이션으로 재계획 트리거
        self.map.set_obstacle([(9, i) for i in range(3, 9)])
        self.capture_frame()
 
        print(f"초기 경로 길이：{path_length}")
        print(f"총 이동 비용：{path_cost:.2f}")
 
        tmp = start
        while tmp != end:
            tmp.set_state("*")  # 경로 표시
            self.capture_frame()
            if tmp.parent.state == "#":  # 경로 노드가 장애물이면 재계획
                self.modify(tmp)
                continue
            tmp = tmp.parent
        tmp.set_state("e")  # 종점 상태 설정
        self.capture_frame()
 
    def modify(self, state):
        # 경로 비용 재수정 및 재계획
        self.modify_cost(state)
        while self.process_state() < state.h:  # 비용 조정 완료까지
            pass
 
    def capture_frame(self):
        # 현재 지도 상태를 애니메이션 프레임으로 저장
        self.frames.append(self.get_frame_array())
 
    def get_frame_array(self):
        # 상태 문자를 색상 코드 행렬로 변환
        color_map = {'.': 0, '#': 1, 's': 2, 'e': 3, '*': 4}
        array = [[color_map.get(self.map.map[i][j].state, 0)
                  for j in range(self.map.col)] for i in range(self.map.row)]
        # 시작점과 종점 강조 표시
        if self.start:
            array[self.start.x][self.start.y] = 2
        if self.end:
            array[self.end.x][self.end.y] = 3
        return array
```



### **6.5그래픽 함수**

```python
# 애니메이션을 그리고 gif로 저장
def animate_map(frames, save_path="dstar_path.gif"):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # 여백 제거
    cmap = ListedColormap([
        '#add8e6',  # 연한 파랑색 - 배경
        '#ff0000',  # 빨강색 - 장애물
        '#0000ff',  # 파랑색 - 시작점
        '#ff1493',  # 빨강색 - 종점 (진한 분홍)
        '#32cd32'   # 초록색 - 경로
    ])
    im = ax.imshow(np.array(frames[0]), cmap=cmap, vmin=0, vmax=4)
    ax.axis('off')
 
    def update(i):
        im.set_array(np.array(frames[i]))
        return [im]
 
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200, blit=True)
    ani.save(save_path, writer='pillow', fps=5)
    print(f"애니메이션이 GIF로 저장됨：{save_path}")
    plt.close(fig)
```



### **6.6주요 함수**

```python
# 메인 함수: 지도 구성, 장애물 설정, D* 실행, 애니메이션 생성
if __name__ == '__main__':
    m = Map(20, 20)
    m.set_obstacle([(2, 1),(2, 2),(2, 3),(2, 4),(4, 3), (4, 4), (4, 5), (4, 6),(5, 3), (6, 3), (7, 3)])  # 장애물 설정
    start = m.map[1][1]  # 시작점 설정
    end = m.map[17][11]  # 종점 설정
    dstar = Dstar(m)
    dstar.run(start, end)  # 경로 계획
    animate_map(dstar.frames, save_path="dstar_path.gif")  # 애니메이션 저장
```



###  6.7실행 효과(코드 좀 바꾼 후)


![image](https://github.com/hachiman888/2025-AIX-DL-Project/blob/main/img/dstar_path.gif)


## 7. 결론

D* 알고리즘의 등장은 A* 알고리즘의 동적 환경에서의 한계를 극복하기 위한 것으로, 지도 변화 발생 시 국부 갱신을 통해 계산 자원을 절약하고 실시간으로 경로를 최적화할 수 있습니다. 이러한 점진적 갱신 메커니즘을 통해 D* 알고리즘은 자율 주행, 로봇 내비게이션, 드론 비행 등 다양한 분야에서 광범위하게 응용되었으며, 동적 경로 계획 기술 발전에 중요한 기여를 하고 있습니다.



**구성원 역할 분담**

* 장봉호옥: 자료 수집
* 증준상: 문서 작성





## 참고

<https://en.wikipedia.org/wiki/A*_search_algorithm>

https://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html

https://cloud.tencent.com/developer/article/2536646

<https://en.wikipedia.org/wiki/D*>

https://web.mit.edu/16.412j/www/html/papers/original_dstar_icra94.pdf

https://qiangbo-workspace.oss-cn-shanghai.aliyuncs.com/2019-02-05-a-star-algorithm/Field%20D%2A-%20An%20Interpolation-based%20Path%20Planner%20and%20Replanner.pdf
