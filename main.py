"""
Card Consulting Chatbot - Main Entry Point
"""
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.chatbot import FreeCardConsultingChatbot
from src.config import CSV_FILE


def main():
    print("\n" + "="*70)
    print("카드 상담 챗봇 시작")
    print("="*70 + "\n")
    
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"데이터 파일 로드 완료: {CSV_FILE}\n")
    except FileNotFoundError:
        print(f"오류: 데이터 파일을 찾을 수 없습니다: {CSV_FILE}")
        print("data/ 폴더에 final_rag_data.csv 파일이 있는지 확인하세요.")
        return
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return
    
    try:
        chatbot = FreeCardConsultingChatbot(df)
    except Exception as e:
        print(f"챗봇 초기화 중 오류 발생: {e}")
        return
    
    print("\n=== 카드 상담 챗봇 (FINAL DEBUG MODE) ===")
    print("명령어: 'quit', 'exit', 'q', '종료' - 챗봇 종료")
    print("       'clear', '초기화' - 대화 기록 초기화\n")
    
    while True:
        try:
            user_input = input("고객: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["quit", "exit", "q", "종료"]:
                print("\n상담을 종료합니다. 감사합니다!")
                break
            
            if user_input.lower() in ["clear", "초기화"]:
                chatbot.chat_history = []
                chatbot.summary_memory = "이전 대화 기록 없음."
                print("\n대화 기록 초기화 완료.\n")
                continue
            
            result = chatbot.chat(user_input)
            
            print(f"\n상담사: {result['answer']}\n")
            
            if result["sources"]:
                print("참조 매뉴얼:")
                for i, src in enumerate(result["sources"][:2], 1):
                    print(f"  {i}. {src['title']} ({src['category']})")
                print()
                
        except KeyboardInterrupt:
            print("\n\n상담을 종료합니다. 감사합니다!")
            break
        except Exception as e:
            print(f"\n오류 발생: {e}\n")
            continue


if __name__ == "__main__":
    main()
