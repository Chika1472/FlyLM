FlyLM-2 SentencePiece 혼합 동역학 추론 번들
model.pt, tokenizer.model, graph/, flylm_inference.py를 같은 폴더에 유지하세요.
이 번들의 모듈로 load_model(model.pt)를 호출한 뒤 reply(model, 질문)를 사용하세요.
필요 패키지: numpy, torch, sentencepiece==0.2.2. 바이트 모델용 추론 모듈은 사용할 수 없습니다.
출력 길이는 max_new_tokens로 지정합니다. 토크나이저 파일은 재학습하거나 교체하지 마세요.
신경 상태는 [4,B,N]: 막전위, 시냅스 신호, 적응, 불응기입니다.
graph/dynamics_evidence.json에 문헌 근거와 전이 가정을 저장했습니다.
미확인 뉴런은 연속 기본값이며 비스파이킹이 확인된 뉴런이 아닙니다.
