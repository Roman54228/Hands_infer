# from pythonosc import udp_client
# from pprint import pprint

# from time import sleep


# class Sender():
#     def __init__(self, ip="127.0.0.1", port=5005):
#         self.client = udp_client.SimpleUDPClient(ip, port)


#     def send_hands(self, hands_data, GMHD_hands):
#         # pprint(hands_data.handedness)
#         # print(GMHD_joints)
#         for i, hand in enumerate(hands_data.handedness):
#             # print(hand.index, hand.display_name)
            
            
#             for j, landmark in enumerate(hands_data.hand_landmarks[i]):
#                 # print(z_pos_est[j].z)
#                 self.send(f"/hands/{hand[0].display_name}/0/{j}", [j, landmark.x, landmark.y, GMHD_hands[i].joints[j].z])
#             print("\n")

#         print("----------------------------------------------------------------------------------------")



#     def send(self, address: str = "/hadns/right/person_number/handmark_number", data: list = []):
#         pprint(data)
#         self.client.send_message(address, data)

# if __name__ == "__main__":
#     sender = Sender()
#     while True:
#         sender.send()
#         sleep(1)




import logging

from pythonosc import udp_client


class Sender():
    def __init__(self, ip="127.0.0.1", port=5005, logging_level="DEBUG"):
        # setup OSC client
        self._client = udp_client.SimpleUDPClient(ip, port)
        
        # setup logger
        self.logger = logging.getLogger("osc_sender")
        
        match logging_level:
            case "DEBUG":
                self.logger.setLevel(logging.DEBUG)
            case "INFO":
                self.logger.setLevel(logging.INFO)
            case "WARN":
                self.logger.setLevel(logging.WARN)
            case "ERROR":
                self.logger.setLevel(logging.ERROR)
            case "CRITICAL":
                self.logger.setLevel(logging.CRITICAL)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(name)s [%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.logger.info(f"OSC Sender with ip {ip} and port {port} is ready")

    def send_hands(self, hands_data, GMHD_hands):
        for i, hand in enumerate(hands_data.handedness):
            for j, landmark in enumerate(hands_data.hand_landmarks[i]):
                try:
                    # TODO: Add hand ID, now we can sand only one right and one right hand. Need additional data?
                    self.send(f"/hands/{hand[0].display_name}/0/{j}", [j, landmark.x, landmark.y, GMHD_hands[i].joints[j].z])

                except Exception as e:
                    self.logger.warning(e)
            
            self.logger.debug("\n")

    def send(self, address: str = "/hadns/right/person_number/handmark_number", data: list = []):
        self.logger.debug(f"{address} {data}")
        self._client.send_message(address, data)







####################     KALMAN      ####################

# import logging
# from pythonosc import udp_client

# ##############################      KALMAN
# from kalmanfilter import KalmanFilter3D
# kf = KalmanFilter3D(process_noise=0.000001, measurement_noise=0.000001, dt=0.2)
# ##############################      KALMAN

# class Sender():
#     def __init__(self, ip="127.0.0.1", port=5005, logging_level="DEBUG"):
#         # setup OSC client
#         self._client = udp_client.SimpleUDPClient(ip, port)
        
#         # setup logger
#         self.logger = logging.getLogger("osc_sender")
        
#         match logging_level:
#             case "DEBUG":
#                 self.logger.setLevel(logging.DEBUG)
#             case "INFO":
#                 self.logger.setLevel(logging.INFO)
#             case "WARN":
#                 self.logger.setLevel(logging.WARN)
#             case "ERROR":
#                 self.logger.setLevel(logging.ERROR)
#             case "CRITICAL":
#                 self.logger.setLevel(logging.CRITICAL)

#         ch = logging.StreamHandler()
#         ch.setLevel(logging.DEBUG)
#         formatter = logging.Formatter('%(asctime)s | %(name)s [%(levelname)s] %(message)s')
#         ch.setFormatter(formatter)
#         self.logger.addHandler(ch)

#         self.logger.info(f"OSC Sender with ip {ip} and port {port} is ready")

#     def send_hands(self, hands_data, GMHD_hands):
#         for i, hand in enumerate(hands_data.handedness):
#             for j, landmark in enumerate(hands_data.hand_landmarks[i]):
#                 try:
#                     ##############################      KALMAN
#                     predicted = kf.predict(landmark.x, landmark.y, GMHD_hands[i].joints[j].z)
#                     predicted_x = predicted[0]
#                     predicted_y = predicted[1]
#                     predicted_z = predicted[2]
#                     ##############################      KALMAN


#                     # TODO: Add hand ID, now we can sand only one right and one right hand. Need additional data?
#                     # self.send(f"/hands/{hand[0].display_name}/0/{j}", [j, landmark.x, landmark.y, GMHD_hands[i].joints[j].z])

#                     ##############################      KALMAN
#                     self.send(f"/hands/{hand[0].display_name}/0/{j}", [j, predicted_x, predicted_y, predicted_z])
#                     ##############################      KALMAN

#                 except Exception as e:
#                     self.logger.warning(e)
            
#             self.logger.debug("\n")

#     def send(self, address: str = "/hadns/right/person_number/handmark_number", data: list = []):
#         self.logger.debug(f"{address} {data}")
#         self._client.send_message(address, data)