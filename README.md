# Tem_fa00-fin-NAME
- 삼정 Future Academy 최종 프로젝트 템플릿입니다. 
---------------------------------------

# 프로젝트 계획서

## 1. 프로젝트 개요
- **프로젝트명** : [프로젝트명]
- **목표** : 구매 이력 기반 상품 추천
- **기간** : 2025년 1월 - 2025년 12월

## 2. 프로젝트 일정
- **분석 및 설계** : 2025년 1월 - 3월
- **개발** : 2025년 4월 - 8월
- **테스트 및 배포** : 2025년 9월 - 12월

## 3. 팀 구성
- **프로젝트 매니저** : A
- **데이터 모델링** : B, C
- **Product 개발** : D
   
---------------------------------------

# 요구사항 정의서

## 1. 기능 요구사항
- [ ] 데이터 수집 기능: [수집 대상 및 방식]
- [ ] 데이터 전처리 기능: [결측치 처리, 이상치 제거 등]
- [ ] 분석 기능: [사용할 알고리즘 또는 분석 기법]
- [ ] 시각화 기능: [대시보드, 차트, 그래프]

## 2. 비기능 요구사항
- [ ] 시스템 안정성: 데이터 처리 시 오류 발생 최소화
- [ ] 성능: 데이터 처리 및 분석 시간 최소화
- [ ] 확장성: 새로운 데이터 추가 및 확장 가능

----------------------------------------

# WBS
## 1. 기획
1.1. 문제 정의  
1.2. 요구사항 수집 및 정의
1.3. 아키텍처 설계
1.4. 프로덕트 디자인

## 2. 데이터 수집, 저장, 전처리
2.1. 데이터 소스 조사  
2.2. 데이터베이스 설계 및 구축
2.2. 데이터 수집 및 저장  
2.3. 데이터 전처리

## 3. 데이터 분석 및 모델링
3.1. 데이터 탐색 및 시각화  
3.2. 모델 선택 및 학습  
3.3. 성능 평가  

## 4. 시스템 개발
4.1. 백엔드 개발
4.2. 프론트엔드 개발
4.3. QA

## 5. 프로덕트 고도화
5.1. 데이터 모델 고도화
5.2. UX 고도화
5.3. QA

## 6. 결과 도출 및 보고
6.1. 결과 요약  
6.2. 보고서 작성  
6.3. 최종 발표

-----------------------------------------

# 모델 정의서

## 1. 데이터 모델
- **사용자 테이블**
  - `user_id` (Primary Key)
  - `username`
  - `password`
  - `email`
  - `role` (admin, user)

- **분석 결과 테이블**
  - `result_id` (Primary Key)
  - `user_id` (Foreign Key)
  - `result_data` (JSON)

## 2. 객체 모델
- **사용자 객체**
  - 속성: `username`, `password`, `email`
  - 메소드: `login()`, `logout()`, `register()`
 
----------------------------------------

# 최종 보고서

## 1. 프로젝트 개요
- **목표**: 구매 이력 기반 상품 추천 시스템 개발
- **기간**: 2025년 1월 - 2025년 12월

## 2. 주요 성과
- 클릭률 0.12%p 개선
- 고객 세그먼트 분류 개선

## 3. 향후 개선 사항
- 사용자 인터페이스 개선
- 성능 최적화 및 리소스 효율성 향상

-----------------------------------------

# 회고

## 1. 잘된 점
- **협업**: 팀원 간의 협업이 원활하게 이루어졌으며, 주기적인 피드백 세션이 유효했다.
- **일정 관리**: 프로젝트 일정에 맞춰 개발이 진행되었고, 큰 지연 없이 배포를 완료했다.

## 2. 개선할 점
- **초기 요구사항 정의 부족**: 초기 요구사항이 부족하여 개발 도중 변경 사항이 많았다.

## 3. 교훈
- **명확한 요구사항 정의**: 초기 단계에서 요구사항을 명확히 정리하는 것이 중요하다.
