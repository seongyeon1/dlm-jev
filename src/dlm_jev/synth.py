"""Synthetic customer-support tickets with ground truth fixed by construction.

Each ticket is sampled as latent attributes first; the labels are pure functions of
those attributes (``labels``), and the Korean text is rendered from phrase banks
afterwards (``render``). The text never decides a label, so the answers are exact.
Hard cases are deliberate: "취소" in a non-churn sense, billing words inside a
how-to question, angry tone on a low-urgency ticket.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from dlm_jev.schema import ChoiceQuestion, DecideRequest, NoulQuestion, Question, ScoreQuestion

CATEGORIES = ["결제·청구", "버그", "계정 접근", "기능 요청", "사용 방법", "서비스 장애"]
URGENCY = ["낮음", "보통", "높음", "긴급"]

QUESTIONS: Dict[str, Question] = {
    "category": ChoiceQuestion(instructions="이 티켓은 어느 분류에 속하는가?", options=CATEGORIES),
    "urgency": ScoreQuestion(
        instructions=(
            "지원팀이 얼마나 빨리 대응해야 하는가? 말투가 아니라 업무 영향으로만 판단한다. "
            "긴급: 10명 이상이 업무를 전혀 못 함. "
            "높음: 우회 방법 없이 막힘, 관리자 계정이 막힘, 잘못 결제된 돈이 있음. "
            "보통: 우회 방법이 있거나 개인 계정 하나만 막힘, 결제 정보 변경 문의. "
            "낮음: 사용 방법 문의, 기능 요청."
        ),
        levels=URGENCY,
    ),
    "churn": NoulQuestion(instructions="고객이 서비스 해지·계약 종료·경쟁사 이전 의사를 밝혔다."),
    "engineering": NoulQuestion(
        instructions=(
            "지금 제품이 깨져 있어서 엔지니어링 팀이 코드 수정·인프라 조치를 해야 해결된다(버그, 서비스 장애). "
            "기능 요청, 결제·청구, 계정 접근, 사용 방법 문의는 지원팀이 처리하므로 해당하지 않는다."
        )
    ),
}


@dataclass(frozen=True)
class Ticket:
    """Latent attributes of one ticket."""

    category: int
    users_affected: int  # 1 = only the writer
    workaround: bool
    admin: bool  # the blocked/affected person is an org admin
    wrong_charge: bool  # billing: money actually taken wrongly
    churn: bool  # explicit intent to leave
    tone: str  # calm | frustrated | angry
    decoy_cancel: bool  # "취소" used in a non-churn sense
    decoy_billing_words: bool  # billing vocabulary in a non-billing ticket


def labels(t: Ticket) -> Dict[str, int]:
    """Ground-truth label index per question id (noul: 0 = yes, 1 = no)."""
    cat = CATEGORIES[t.category]
    if cat == "서비스 장애":
        urgency = 3 if t.users_affected >= 10 else 2
    elif cat == "버그":
        urgency = 1 if t.workaround else (3 if t.users_affected >= 10 else 2)
    elif cat == "계정 접근":
        urgency = 2 if t.admin or t.users_affected >= 10 else 1
    elif cat == "결제·청구":
        urgency = 2 if t.wrong_charge else 1
    else:  # 기능 요청, 사용 방법
        urgency = 0
    engineering = cat in ("버그", "서비스 장애")
    return {
        "category": t.category,
        "urgency": urgency,
        "churn": 0 if t.churn else 1,
        "engineering": 0 if engineering else 1,
    }


def sample(rng: random.Random) -> Ticket:
    """Draw one ticket's attributes."""
    category = rng.randrange(len(CATEGORIES))
    cat = CATEGORIES[category]
    users = rng.choice([1, 1, 1, 3, 5, 12, 40, 200]) if cat in ("버그", "서비스 장애", "계정 접근") else 1
    return Ticket(
        category=category,
        users_affected=users,
        workaround=cat == "버그" and rng.random() < 0.5,
        admin=cat == "계정 접근" and rng.random() < 0.4,
        wrong_charge=cat == "결제·청구" and rng.random() < 0.5,
        churn=rng.random() < 0.25,
        tone=rng.choice(["calm", "calm", "frustrated", "angry"]),
        decoy_cancel=rng.random() < 0.25,
        decoy_billing_words=cat != "결제·청구" and rng.random() < 0.2,
    )


_PRODUCTS = ["대시보드", "리포트 내보내기", "알림 설정", "API 연동", "팀 관리 화면", "모바일 앱", "검색 기능"]
_GREET = ["안녕하세요.", "안녕하세요, 담당자님.", "수고 많으십니다.", "", "문의드립니다."]
_SIGN = ["확인 부탁드립니다.", "빠른 답변 부탁드려요.", "감사합니다.", "회신 기다리겠습니다.", ""]
_TONE = {
    "calm": ["", "편하실 때 봐주세요."],
    "frustrated": ["벌써 두 번째 문의인데 아직 해결이 안 됐네요.", "솔직히 좀 답답합니다."],
    "angry": ["이게 말이 됩니까? 정말 화가 납니다.", "도대체 일을 어떻게 하시는 건지 모르겠네요!!"],
}


def _who(t: Ticket, rng: random.Random) -> str:
    if t.users_affected == 1:
        return "저만"
    if t.users_affected < 10:
        return f"저희 팀 {t.users_affected}명이"
    return rng.choice([f"회사 전체 {t.users_affected}명이", f"전 직원({t.users_affected}명)이", "모든 사용자가"])


def _body(t: Ticket, rng: random.Random) -> List[str]:
    cat = CATEGORIES[t.category]
    product = rng.choice(_PRODUCTS)
    if cat == "결제·청구":
        if t.wrong_charge:
            return [
                rng.choice(["이번 달 요금이 두 번 결제됐습니다.", "해지한 좌석 요금이 계속 청구되고 있어요."]),
                "카드 명세서에 중복 승인이 찍혀 있습니다.",
            ]
        return [rng.choice(["세금계산서 발행 주소를 바꾸고 싶습니다.", "다음 달부터 연간 결제로 바꿀 수 있나요?"])]
    if cat == "버그":
        lines = [f"{product}에서 저장을 누르면 오류 메시지가 뜹니다.", f"{_who(t, rng)} 같은 증상을 겪고 있어요."]
        lines.append(
            "새로고침하고 다시 하면 되긴 합니다." if t.workaround else "다른 방법으로도 전혀 진행이 안 됩니다."
        )
        return lines
    if cat == "계정 접근":
        who = "관리자 계정" if t.admin else "제 계정"
        return [
            f"{who}으로 로그인이 안 됩니다.",
            "비밀번호 재설정 메일도 오지 않아요.",
            f"{_who(t, rng)} 접속을 못 하고 있습니다.",
        ]
    if cat == "기능 요청":
        return [f"{product}에 엑셀 일괄 업로드 기능이 있으면 좋겠습니다.", "검토해 주실 수 있을까요?"]
    if cat == "사용 방법":
        return [f"{product}에서 필터를 저장하는 방법을 알려주세요.", "도움말을 찾아봤는데 잘 모르겠네요."]
    return [
        rng.choice(["서비스 접속 자체가 안 됩니다.", "화면이 계속 503 에러만 나옵니다."]),
        f"{_who(t, rng)} 지금 업무를 못 하고 있어요.",
        "상태 페이지에는 정상이라고 나오는데요.",
    ]


def render(t: Ticket, rng: random.Random) -> str:
    """Korean ticket text for the attributes; labels never depend on the wording chosen here."""
    lines = [rng.choice(_GREET), *_body(t, rng)]
    if t.decoy_billing_words:
        lines.append(rng.choice(["참고로 결제는 문제없이 잘 되고 있습니다.", "청구서 쪽은 이상 없어요."]))
    if t.decoy_cancel:
        lines.append(
            rng.choice(["진행 중이던 내보내기 작업은 제가 취소해 두었습니다.", "어제 예약한 회의는 취소했어요."])
        )
    lines.extend(rng.sample(_TONE[t.tone], 1))
    if t.churn:
        lines.append(
            rng.choice(
                [
                    "이번 주 안에 해결 안 되면 계약 해지하겠습니다.",
                    "다른 서비스로 옮기는 것도 검토하고 있습니다.",
                    "재계약은 어려울 것 같네요.",
                ]
            )
        )
    lines.append(rng.choice(_SIGN))
    return " ".join(x for x in lines if x)


def _who_shifted(t: Ticket) -> str:
    if t.users_affected == 1:
        return "제 쪽만 그런 것 같고"
    if t.users_affected < 10:
        return f"옆자리 동료들 포함해서 {t.users_affected}명 정도"
    return f"사내 {t.users_affected}명 전부"


def _body_shifted(t: Ticket, rng: random.Random) -> List[str]:
    cat = CATEGORIES[t.category]
    product = rng.choice(["캘린더 동기화", "파일 공유", "워크플로 자동화", "SSO 로그인 페이지", "통계 위젯"])
    if cat == "결제·청구":
        if t.wrong_charge:
            return [
                rng.choice(
                    [
                        "지난주에 플랜 다운그레이드했는데 비싼 요금이 그대로 빠져나갔네요.",
                        "환불 처리됐다던 금액이 다시 인출됐습니다.",
                    ]
                )
            ]
        return [
            rng.choice(["영수증에 사업자번호 넣어서 다시 받을 수 있을까요?", "결제 카드를 법인카드로 바꾸려는데요."])
        ]
    if cat == "버그":
        way = "다른 브라우저로 하면 우회는 됩니다." if t.workaround else "어떤 방법을 써도 작업이 안 넘어가요."
        return [f"{product} 누르면 빈 화면만 나오고 멈춰요.", f"{_who_shifted(t)} 이러네요.", way]
    if cat == "계정 접근":
        who = "워크스페이스 오너 계정" if t.admin else "개인 계정"
        return [
            f"{who}이 잠겼다고 나옵니다.",
            "2단계 인증 코드가 안 맞는다고 해요.",
            f"{_who_shifted(t)} 못 들어가는 상황입니다.",
        ]
    if cat == "기능 요청":
        return [f"{product}에 다크 모드도 지원되면 좋겠어요.", "로드맵에 있는지 궁금합니다."]
    if cat == "사용 방법":
        return [f"{product} 권한을 팀원별로 다르게 주려면 어디서 설정하나요?"]
    return [
        rng.choice(["지금 로그인 페이지부터 타임아웃 나요.", "API 호출이 전부 500으로 떨어집니다."]),
        f"{_who_shifted(t)} 손 놓고 있어요.",
    ]


def render_shifted(t: Ticket, rng: random.Random) -> str:
    """Same attributes and labels, but none of ``render``'s phrases: a distribution-shift test set."""
    lines = [
        rng.choice(["혹시 확인 가능하실까요?", "급하게 여쭤봐요.", "", "고객센터 앞", "Hi 팀,"]),
        *_body_shifted(t, rng),
    ]
    if t.decoy_billing_words:
        lines.append("요금 청구는 정상적으로 되고 있고요.")
    if t.decoy_cancel:
        lines.append(rng.choice(["아까 올린 업로드는 중간에 취소했습니다.", "알림 구독은 제가 해지해 뒀어요."]))
    tone = {"calm": "", "frustrated": "같은 얘기 몇 번째인지 모르겠네요.", "angry": "진짜 너무하시네요, 장난합니까?"}
    lines.append(tone[t.tone])
    if t.churn:
        lines.append(
            rng.choice(
                [
                    "경쟁 제품 도입 품의 올렸습니다.",
                    "다음 분기부터는 안 쓸 생각입니다.",
                    "위약금 알려주세요, 정리하려고요.",
                ]
            )
        )
    lines.append(rng.choice(["", "부탁드립니다.", "ㅠㅠ"]))
    return " ".join(x for x in lines if x)


def make_dataset(n: int, seed: int, shifted: bool = False) -> List[Tuple[DecideRequest, Dict[str, int], Ticket]]:
    """``n`` four-question requests with their ground-truth labels and latent attributes."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        t = sample(rng)
        text = render_shifted(t, rng) if shifted else render(t, rng)
        out.append((DecideRequest(state=text, questions=dict(QUESTIONS)), labels(t), t))
    return out
