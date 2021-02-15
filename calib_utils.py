import cv2
import numpy as np


class CamCalibration:
    def __init__(self, board_size, board_width, board_height):
        self.board_size = board_size
        self.board_width = board_width
        self.board_height = board_height

    def findCorners(self, video_path):
        '''
        video_path:   path of calibration video 
        corners_2d: [num_frames, num_corners, 2]
        frame_list:   [num_frames, heigh, width, 3]
        '''
        corners_2d, frame_list = [], []
        cap = cv2.VideoCapture(video_path)
        while(True):
            succ, frame = cap.read()
            if succ == False:
                break
            
            # Find chessboard corners
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.board_width, self.board_height))
            if ret == True:
                corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                corners_2d.append(corners[:,0,:])  # corners.shape = [num_corners, 1, 2]
                frame_list.append(frame)
        
        cap.release()
        corners_2d, frame_list = np.array(corners_2d), np.array(frame_list)
        return corners_2d, frame_list

    def inCalibration(self, video_path, check=False):
        # Find corners in pixel coordinates
        corners_2d, frame_list = self.findCorners(video_path)

        # Params
        num_frames = len(frame_list)
        assert num_frames > 0, "No corners found"
        print('num of images with corners: {}'.format(num_frames))

        # Generate corners in world coordinates
        corners_3d = []
        for _ in range(num_frames):
            corners = []
            for i in range(self.board_height):
                for j in range(self.board_width):
                    corners.append([i*self.board_size, j*self.board_size, 0])
            corners_3d.append(corners)
        corners_3d = np.array(corners_3d).astype('float32')
        
        # Intrinsic calibration
        period = max(num_frames // 40, 1)
        img_height, img_width = frame_list.shape[1], frame_list.shape[2]
        corners_2d, corners_3d, frame_list = corners_2d[::period], corners_3d[::period], frame_list[::period]
        print('corners_2d.shape = {}, corners_3d.shape = {}, frame_list.shape = {}'.format(corners_2d.shape, corners_3d.shape, frame_list.shape))

        cameraMatrix, distCoeffs = None, None
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(corners_3d, corners_2d, (img_height, img_width), cameraMatrix, distCoeffs)

        print('the overall RMS re-projection error = {}'.format(retval))

        # Check
        if check == True:
            for i in range(len(frame_list)):
                frame = frame_list[i]
                corners, _ = cv2.projectPoints(corners_3d[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
                # err = cv2.norm(pixel_corners[i], tmp_pixel_corners, cv2.NORM_L2)/len(tmp_pixel_corners)
                # errors.append(err)

                cv2.drawChessboardCorners(frame, (self.board_width, self.board_height), corners, True)
                cv2.imshow('test', frame)
                if cv2.waitKey(500) == 27:
                    break
            cv2.destroyAllWindows()

        return cameraMatrix, distCoeffs


    def exCalibration(self, img, M, dists):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.board_width, self.board_height))
        if ret == True:
            corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        cv2.drawChessboardCorners(img, (self.board_width, self.board_height), corners, True)
        cv2.imshow('test', img)
        cv2.waitKey(3000)

        corners_2d = corners[:,0,:].astype('float32')  # corners.shape = [num_corners, 1, 2]

        corners_3d = []
        for i in range(self.board_height):
            for j in range(self.board_width):
                corners_3d.append([i*self.board_size, j*self.board_size, 0])
        corners_3d = np.array(corners_3d).astype('float32')

        print(corners_3d.shape, corners_2d.shape, M.shape, dists.shape)

        rvec, tvec = None, None
        retval, rvec, tvec = cv2.solvePnP(corners_3d, corners_2d, M, dists, rvec, tvec)
        print('solvePnP error: {}'.format(retval))

        R, _ = cv2.Rodrigues(rvec)

        return R, rvec, tvec

'''
    def stereoCalibration(self, corners1_2d, corners2_2d, M1, dists1, M2, dists2):
        corners_3d = []
        for i in range(self.board_height):
            for j in range(self.board_width):
                corners_3d.append([i*self.board_size, j*self.board_size, 0])
        corners_3d = np.array([corners_3d]).astype('float32')

        corners1_2d, corners2_2d = np.array([corners1_2d]).astype('float32'), np.array([corners2_2d]).astype('float32')
        print(corners1_2d.shape, corners2_2d.shape, corners_3d.shape)

        # corners_3d, corners1_2d, corners2_2d = np.zeros((1,48,3)).astype('float32'), np.zeros((1,48,2)), np.zeros((1,48,2))
        # print(corners_3d.shape)

        retval, M1, dists1, M2, dists2, R, T, E, F = cv2.stereoCalibrate(corners_3d, corners1_2d, corners2_2d, M1, dists1, M2, dists2, (1080,1920))
        print(retval)
        print(R)
        print(T)

        return R, T
'''