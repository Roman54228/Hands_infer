from hand_detection_app import HandDetectionApp


def main():
    """Главная функция приложения"""
    # Конфигурация путей к моделям
    yolo_model_path = "/home/cineai/ViduSdk/python/TRT_Roma/newone/upside.engine"
    hand_gesture_model_path = "my_model.engine"
    kps_model_path = "cls_kps.engine"
    
    # Конфигурация OSC
    osc_ip = "10.0.0.101"
    osc_port = 5055
    
    # Создание и запуск приложения
    app = HandDetectionApp(
        yolo_model_path=yolo_model_path,
        hand_gesture_model_path=hand_gesture_model_path,
        kps_model_path=kps_model_path,
        osc_ip=osc_ip,
        osc_port=osc_port,
        stream_index=1
    )
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
