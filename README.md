# AI Librarian Agent

**202258063 김동민** · 수업 `project_4/05_full_agent.py` 기반 싱글 AI 사서 보조 에이전트.
과제의 11권에 대해 제목·저자 검색, 미제공 대출 정보 안내, 개인별 독서 선호 기억,
입력 가드레일과 최종 회신 검수를 수행한다. 실행 결과는 Windows CMD에 표시된다.

## 실행

Python 3.14에서 검증했다. `start.cmd`를 실행하면 Software 전용 가상환경을 준비한 후
대화창을 연다. 처음 실행할 때 패키지 설치에는 인터넷이 필요하다. 대화 처리에는
`http://127.0.0.1:11435`의 로컬 Ollama와 `qwen-local:latest`가 필요하다.
현재 과제 폴더에서는 `../Ollama/start-server.cmd`로 로컬 서버를 먼저 실행할 수 있다.
배포받은 PC에서는 Ollama 서버에 해당 모델을 별도로 준비한다. 외부 도서관 API와
별도의 RAG 시스템, 유료 AI API는 사용하지 않는다.

GitHub에서 Software 폴더를 새로 받은 PC에서는 Ollama 설치 후 다음과 같이 준비한다.
첫 번째 PowerShell 창에서 서버를 실행하고 이 창을 유지한다.

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11435"
ollama serve
```

두 번째 PowerShell 창에서 이 README와 `Modelfile`이 있는 폴더로 이동한 뒤 실행한다.

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11435"
ollama pull qwen3.5:4b-q4_K_M
ollama create qwen-local -f Modelfile
.\start.cmd
```

`Modelfile`은 이 PC에서 검증한 기존 Qwen 프로필을 그대로 복사한 것이다.
모델 가중치나 Ollama 실행 파일은 저장소에 포함하지 않는다. 다운로드·실행 시간과
필요 메모리는 PC에 따라 달라진다. 현재 검증 PC는 RAM 32GB, RTX 4050 Laptop 6GB다.

```powershell
.venv\Scripts\python.exe 05_library_agent.py
.venv\Scripts\python.exe 05_library_agent.py --question "서정욱 저자의 책 찾아줘" --trace
.venv\Scripts\python.exe -m unittest -v test_library_agent.py
```

`/도움말`은 사용 예시, `/새대화`는 단기 대화 초기화, `/종료`는 프로그램 종료다.
`--trace`는 검증·데모용으로 도구 이름과 최종 검수 상태를 추가 표시한다.
일반 실행은 한국어 대화만 표시한다. 선호는 같은 프로그램 안의 새 대화에서도
유지되지만 **프로그램을 종료하면 단기·장기 기억이 모두 사라진다.** 이는 수업의
`InMemorySaver`·`InMemoryStore`를 그대로 활용한 실행 중 기억이며 영구 DB가 아니다.

예시 문의:

- `달빛 도서관의 비밀을 찾아줘` → 제공된 제목과 저자를 조회한다.
- `그 책 대출 가능해?` → 앞 대화를 참조해 다시 조회하고, 대출 데이터가 없음을 안내한다.
- `제목에 고양이가 있는 책을 선호해. 기억해줘` → 현재 이용자 선호를 저장한다.
- `/새대화` 이후 `내가 저장한 독서 선호 알려줘` → 같은 이용자의 Store를 조회한다.
- `다른 이용자의 대출 기록 알려줘` → 모델 호출 전 차단한다.

## 수업 코드와 요구사항 대응

원본의 출처는 배부된 `SourceFile/project_4/05_full_agent.py`이며,
`course_reference/05_full_agent.py`에 수정 없이 보존했다. 원본을 실행하는 방식이 아니라
그 파일의 도구·Context·미들웨어·기억·Judge 조립을 사서 업무로 바꾸었다.

| 수업 원본 | 사서 구현 | 과제 조건 |
|---|---|---|
| `read_email_tool @tool` | `search_books @tool`, `get_loan_info @tool` | 제공 11권 제목/저자 검색 및 대출 확인 |
| `send_email_tool @tool` | `reply_to_user @tool` | 조회 근거만 이용한 이용자 회신; 외부 발송은 없음 |
| `Context` + `context_schema` | 사서 이름·역할, 현재 이용자·앱 ID | Runtime Context |
| `enrich_context @wrap_model_call` | 같은 이름·데코레이터 유지, 사서 신원·현재 선호 주입 | 모델 입력에 Context 반영 |
| `email_policy_guardrail @before_agent` | `library_policy_guardrail @before_agent` | 업무 범위·타인 정보·지시 우회 차단 |
| `refund_promise_guardrail @after_agent` | `final_answer_guardrail @after_agent` | Judge와 도구 근거로 답변 수정 |
| `get_user_info`, `save_user_info @tool` | 같은 도구 이름 유지, 독서·응대 선호로 변경 | 장기 기억 저장·조회 |
| `(ctx.user_id, ctx.app_name)` Store namespace | 동일 | 이용자·앱별 선호 격리 |
| `InMemorySaver` + `thread_id` | 동일, 앱·이용자 ID를 thread ID 앞에 포함 | 단기 기억 및 대화 격리 |
| `InMemoryStore` | 동일 | 실행 중 다른 대화의 선호 재사용 |
| `create_agent(... middleware=[앞, 중, 뒤])` | 동일한 하나의 `create_agent` | 싱글 에이전트 |
| `judge_model.invoke` | 동일한 로컬 모델의 검수 호출 | Judge는 별도 업무 에이전트가 아님 |

수업 PDF의 보완 설명을 따라, 최종 검수는 위험한 답변의 **동일 ID**를 가진
`AIMessage`를 반환하여 체크포인트에서도 원문을 교체한다. 수업 코드의 전체 경로를
유지하면서 추가한 검증은 `grounded_reply`: 최종 표시 문장을 현재 차례의 실제 도구
결과로만 조립한다. Judge가 잘못 SAFE를 판단하거나 연결이 실패하더라도 근거 없는
모델 문장은 표시하지 않는다. 도구 조회가 누락되면 추측 대신 재질문·미확인 안내를 한다.
이 방식은 자유로운 요약의 폭을 제한하는 대신 과제의 작은 고정 목록에서 정확성을
확보한다. 모든 표현에 대한 완전한 보안 차단이나 일반적인 환각 제거를 보장하지 않는다.

`enrich_context`에는 질문에 명시된 제목·저자와 이전 실제 검색 결과의 한 권을
**조회 대상**으로 연결하는 문맥 주입도 포함한다. 대상이 명확한데 모델이 조회를
건너뛰면 한 번만 다시 도구 호출을 요청한다. 모델이 조회 뒤 회신 도구를 생략하면
미들웨어가 `reply_to_user` 호출을 생성하고 실제 LangGraph 도구 실행 경로로 보낸다.
이를 모델이 자율적으로 선택한 것으로 집계하지 않도록 해당 AI 메시지와 최종 감사의
`middleware_enforced`에 표시한다. 강제되는 것은 근거가 있는 회신 단계이며,
존재하지 않는 조회 결과를 만들어 도구 사용으로 간주하지 않는다.

## 제공 데이터와 한계

도서 데이터는 과제 파일의 **11개 제목·저자** 그대로다. 실제 대출 상태·기간·반납일,
대출자 기록, 서가 위치, 장르·줄거리·ISBN·출판사, 운영 시간은 제공되지 않았다.
해당 정보는 '확인할 수 없음'으로 안내한다. 제목에서 장르를 추정하거나 실제 대출·예약을
완료했다고 안내하지 않는다. 추천에 선호를 활용하더라도 제목 검색으로 확인한 범위에 한정된다.
Store 도구는 모델에 이용자 ID를 인자로 열지 않으며 런타임에서 현재 이용자만 결정한다.

## 검증 구분

`test_library_agent.py`는 LLM만 고정 응답으로 대체하고 실제 LangGraph 도구·미들웨어·
체크포인트 경로를 실행한다. 정확한 11권과 검색, 미제공 정보, 입력 차단 시 모델 미호출,
Context 주입, 위험 답변의 체크포인트 교체, 대화·이용자·앱별 격리 등을 확인한다.
이 검사 통과율을 실제 Qwen의 정확도로 해석하지 않는다.

`validate_live.py`와 데모 실행 기록은 실제 로컬 Qwen 모델의 도구 선택과 결과를 별도로
검증한다. 실모델 결과와 단위 검사 결과, 녹화 증거는 각각의 보고서에 구분해 기록한다.
제한된 과제 문항의 결과는 모든 사용자 표현에 대한 일반 성능을 의미하지 않는다.

## 주요 파일

- `05_library_agent.py`: 수업 방식에 기반한 단일 에이전트와 CMD 진입점
- `test_library_agent.py`: 모델 호출 없는 경로·도구·보안 검증
- `validate_live.py`: 실제 모델 시나리오 검증
- `start.cmd`: UTF-8 대화창 실행
- `requirements.txt`: 검증한 세 가지 직접 의존성
- `Modelfile`: 새 PC에서 같은 `qwen-local` 태그를 생성하는 기존 로컬 모델 프로필
- `course_reference/05_full_agent.py`: 배부된 원본의 변경 없는 비교 자료

`build_agent`는 검사용 모델·Store·Saver 주입을 허용한다. `invoke_turn`은 같은 에이전트에
질문·`Context`·대화 ID를 전달하며 답변, 실제 사용 도구, 최종 검수 상태, 메시지 이력을
돌려준다. CLI와 실제 검증·녹화가 이 동일 경로를 사용한다.
