import librosa
import numpy as np
import soundfile as sf  # librosa는 파일을 쓸 때 soundfile을 사용합니다.
import colorednoise as cn
import os

# --- 4단계: SNR 계산 함수 ---
def calculate_snr(y_original, y_compared):
    """
    두 오디오 신호 간의 SNR(신호 대 잡음비)을 계산합니다.
    :param y_original: 원본 신호 (Ground Truth)
    :param y_compared: 비교 대상 신호 (노이즈가 꼈거나, 디노이징된 신호)
    :return: SNR 값 (dB)
    """
    # STFT/ISTFT 과정에서 길이가 약간 달라질 수 있으므로, 짧은 쪽에 맞춥니다.
    min_len = min(len(y_original), len(y_compared))
    y_original = y_original[:min_len]
    y_compared = y_compared[:min_len]

    # 신호 전력 (Signal Power) 계산
    signal_power = np.sum(y_original ** 2)
    
    # 노이즈 전력 (Noise Power) 계산 (원본 신호와 비교 신호의 차이)
    noise_power = np.sum((y_original - y_compared) ** 2)

    # SNR 계산 (dB 단위)
    if noise_power == 0:
        return np.inf  # 노이즈가 전혀 없으면 (완벽한 복원) SNR은 무한대
        
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# --- 2단계: 베이스라인 모델 (이동 평균 필터) ---
def apply_moving_average(y, window_size=5):
    """
    신호에 이동 평균 필터를 적용합니다.
    :param y: 입력 신호
    :param window_size: 평균을 낼 창(window)의 크기 (홀수 권장)
    :return: 필터가 적용된 신호
    """
    # np.convolve를 사용하여 이동 평균을 간단히 구현
    window = np.ones(window_size) / window_size
    # 'same' 모드를 사용해 원본 신호와 길이를 맞춥니다.
    return np.convolve(y, window, mode='same')

# --- 3단계: 제안 모델 (STFT + 소프트 쓰레시홀딩) ---
# (수정됨) 'noise_profile_sec' 대신 'y_noise_profile' (순수 노이즈 배열)을 직접 받도록 변경
def apply_stft_filter(y_noisy, y_noise_profile, sr, T_gain=1.5):
    """
    STFT와 소프트 쓰레시홀딩을 사용해 노이즈를 제거합니다.
    (수정됨: 'noise_profile_sec' 대신 'y_noise_profile'을 직접 받음)
    
    :param y_noisy: 노이즈가 낀 신호
    :param y_noise_profile: 노이즈 프로파일링에 사용할 순수 노이즈 신호
    :param sr: 샘플 레이트
    :param T_gain: 임계값 T에 곱해줄 보정 계수
    :return: 디노이징된 신호
    """
    
    # 3-2: STFT (단시간 푸리에 변환)
    # y_noisy 전체를 스펙트로그램 S로 변환합니다.
    S_noisy = librosa.stft(y_noisy)
    
    # 3-1: 노이즈 프로파일링 (임계값 T 설정)
    # (수정됨) 0.5초 가정 대신, 전달받은 '순수 노이즈' 신호(y_noise_profile)로
    # 스펙트로그램을 만들어 노이즈 프로필을 계산합니다.
    S_noise_profile = librosa.stft(y_noise_profile)
    
    # 주파수 빈(bin)별로 평균 에너지(크기)를 계산합니다.
    # axis=1 (시간 축)을 기준으로 평균을 냅니다.
    mean_noise_mag = np.mean(np.abs(S_noise_profile), axis=1)
    
    # 임계값 T를 설정합니다. (캔버스 수식)
    # T는 각 주파수 빈(1D)에 대해 계산됩니다.
    T = mean_noise_mag * T_gain

    # 3-3: 필터링 (소프트 쓰레시홀딩)
    # STFT 결과를 크기(Magnitude)와 위상(Phase)으로 분리합니다.
    S_mag, S_phase = librosa.magphase(S_noisy)
    
    # 임계값 T를 스펙트로그램 T (2D)로 확장합니다. (Broadcasting)
    # (n_freq,) -> (n_freq, 1)
    T_expanded = T[:, np.newaxis] 
    
    # 캔버스 수식 (S'_magnitude = max(0, |S| - T))을 그대로 구현합니다.
    # Numpy의 브로드캐스팅 기능으로 S_mag(2D)의 모든 프레임에서 T(1D)가 차감됩니다.
    S_mag_denoised = np.maximum(0, S_mag - T_expanded)
    
    # 3-4: ISTFT (역 STFT)
    # 디노이징된 크기와 원본 위상을 다시 합쳐서 스펙트로그램 S'를 만듭니다.
    S_denoised = S_mag_denoised * S_phase
    
    # S'를 ISTFT를 통해 시간 도메인 신호 y'로 복원합니다.
    y_proposed = librosa.istft(S_denoised)
    
    return y_proposed

# --- 메인 실행 ---
if __name__ == "__main__":
    
    # --- 상수 정의 ---
    ORIGINAL_FILE = "original.wav"
    # 프로젝트 개요 1단계에서 정의한 노이즈 3종류
    NOISE_TYPES = {
        "white": 0,    # White Noise (beta = 0)
        "brown": 2,    # Brown Noise (beta = 2, 1/f^2)
        "violet": -2   # Violet Noise (beta = -2, f^2)
    }
    # 노이즈 강도 (원본 신호 RMS 대비 15% 수준의 RMS를 갖도록 설정)
    TARGET_NOISE_RMS_RATIO = 0.15 
    
    # 3단계에서 사용할 하이퍼파라미터
    # (수정됨) NOISE_PROFILE_DURATION은 더 이상 사용하지 않습니다.
    THRESHOLD_GAIN = 1.5          # T 보정 계수
    
    # 2단계에서 사용할 하이퍼파라미터
    # (수정됨) 5는 너무 작습니다. 101 (홀수) 정도로 늘립니다.
    MA_WINDOW_SIZE = 101            # 이동 평균 창 크기

    # --- 1단계: 원본 신호 로드 ---
    if not os.path.exists(ORIGINAL_FILE):
        print(f"오류: {ORIGINAL_FILE} 파일을 찾을 수 없습니다.")
        print("스크립트와 같은 디렉토리에 original.wav 파일을 위치시켜주세요.")
        exit()

    # 원본 파일 로드. librosa는 기본적으로 모노, sr=22050으로 로드합니다.
    # sr=None으로 설정하면 원본 샘플 레이트를 유지합니다.
    try:
        y_original, sr = librosa.load(ORIGINAL_FILE, sr=None)
        print(f"'{ORIGINAL_FILE}' 로드 완료 (Sample Rate: {sr} Hz)")
    except Exception as e:
        print(f"'{ORIGINAL_FILE}' 파일 로드 중 오류 발생: {e}")
        exit()

    # 최종 SNR 결과를 저장할 딕셔너리
    all_snr_results = {}
    
    print("-" * 30)
    print("디노이징 프로젝트 자동화를 시작합니다.")
    print(f"적용할 노이즈: {list(NOISE_TYPES.keys())}")
    print("-" * 30)


    # --- 각 노이즈 유형별로 전체 파이프라인 반복 ---
    for noise_name, beta in NOISE_TYPES.items():
        print(f"\n[{noise_name.upper()} Noise] 처리 중...")

        # --- 1단계: 실험 데이터 생성 ---
        print("  1단계: 노이즈 생성 및 믹싱 중...")
        # colorednoise 라이브러리를 사용해 노이즈 생성 (beta 값 사용)
        noise_raw = cn.powerlaw_psd_gaussian(beta, len(y_original))
        
        # 원본 신호와 노이즈의 RMS(평균 제곱근)를 계산하여 노이즈 레벨을 일정하게 조절
        signal_rms = np.sqrt(np.mean(y_original ** 2))
        noise_rms = np.sqrt(np.mean(noise_raw ** 2))
        
        # 노이즈의 RMS가 (신호 RMS * 비율)이 되도록 스케일링
        scaled_noise = noise_raw * (signal_rms * TARGET_NOISE_RMS_RATIO) / noise_rms
        
        # 원본 신호에 스케일링된 노이즈를 믹스
        y_noisy = y_original + scaled_noise
        
        # 오디오 클리핑 방지 (소리가 -1.0 ~ 1.0 범위를 벗어나지 않도록)
        y_noisy = np.clip(y_noisy, -1.0, 1.0)
        
        # 파일로 저장 (예: noisy_white.wav)
        output_filename_noisy = f"noisy_{noise_name}.wav"
        sf.write(output_filename_noisy, y_noisy, sr)
        print(f"    -> '{output_filename_noisy}' 저장 완료.")

        # --- 2단계: 베이스라인 모델 적용 ---
        print("  2단계: 베이스라인 (이동 평균) 필터 적용 중...")
        y_baseline = apply_moving_average(y_noisy, window_size=MA_WINDOW_SIZE)
        y_baseline = np.clip(y_baseline, -1.0, 1.0)
        
        output_filename_baseline = f"baseline_result_{noise_name}.wav"
        sf.write(output_filename_baseline, y_baseline, sr)
        print(f"    -> '{output_filename_baseline}' 저장 완료.")

        # --- 3단계: 제안 모델 적용 ---
        print("  3단계: 제안 모델 (STFT + Soft Threshold) 적용 중...")
        # (수정됨) 0.5초 가정(noise_profile_sec) 대신, 1단계에서 생성한
        # 'scaled_noise'를 직접 전달하여 정확한 노이즈 프로필을 사용합니다.
        y_proposed = apply_stft_filter(y_noisy, scaled_noise, sr, 
                                     T_gain=THRESHOLD_GAIN)
        y_proposed = np.clip(y_proposed, -1.0, 1.0)
        
        output_filename_proposed = f"proposed_result_{noise_name}.wav"
        sf.write(output_filename_proposed, y_proposed, sr)
        print(f"    -> '{output_filename_proposed}' 저장 완료.")

        # --- 4단계: SNR 계산 ---
        print("  4단계: SNR 계산 중...")
        # 원본 vs 노이즈
        snr_1 = calculate_snr(y_original, y_noisy)
        # 원본 vs 베이스라인
        snr_2 = calculate_snr(y_original, y_baseline)
        # 원본 vs 제안 모델
        snr_3 = calculate_snr(y_original, y_proposed)
        
        # 결과 저장
        all_snr_results[noise_name] = (snr_1, snr_2, snr_3)
        print("    -> 계산 완료.")

    # --- 최종 결과 출력 ---
    print("\n" + "=" * 40)
    print("         모든 작업 완료 - SNR 비교 결과")
    print("=" * 40)
    
    for noise_name, (snr_1, snr_2, snr_3) in all_snr_results.items():
        print(f"\n--- {noise_name.upper()} Noise 결과 ---")
        print(f"  SNR 1 (Original vs Noisy)     : {snr_1:8.2f} dB")
        print(f"  SNR 2 (Original vs Baseline)  : {snr_2:8.2f} dB")
        print(f"  SNR 3 (Original vs Proposed)  : {snr_3:8.2f} dB")
        
        improvement_baseline = snr_2 - snr_1
        improvement_proposed = snr_3 - snr_1
        
        print(f"    -> Baseline 개선   : {improvement_baseline:+.2f} dB")
        print(f"    -> Proposed 개선   : {improvement_proposed:+.2f} dB")

    print("\n" + "=" * 40)
    print("같은 디렉토리에 생성된 .wav 파일들을 확인해보세요.")

