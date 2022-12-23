//
//  ViewController.swift
//  Example
//
//  Created by Tomoya Hirano on 2020/04/02.
//  Copyright © 2020 Tomoya Hirano. All rights reserved.
//

import UIKit
import AVFoundation
import CoreML

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate, TrackerDelegate {
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var toggleView: UISwitch!
    var previewLayer: AVCaptureVideoPreviewLayer!
    @IBOutlet weak var xyLabel:UILabel!
    @IBOutlet weak var frameTimeLabel:UILabel!
    @IBOutlet weak var inferenceClsLabel:UILabel!
    @IBOutlet weak var featurePoint: UIView!
    let camera = Camera()
    let tracker: HandTracker = HandTracker()!
    let model: saved_model = {
    do {
        let config = MLModelConfiguration()
        return try saved_model(configuration: config)
    } catch {
        print(error)
        fatalError("Couldn't create SleepCalculator")
    }
    }()
    //Classes to assign predictions
    let classes:[String] = ["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
    //Variables to measure the inferences
    var time1 = CACurrentMediaTime();
    var timeOld = CACurrentMediaTime();
    var timeClsBefore = CACurrentMediaTime();
    var timeClsAfter = CACurrentMediaTime();
    var infBuf = Array(repeating: 0, count: 10)
    var timeClsBuf = Array(repeating: 0, count: 10)
    var infCounter = 0
    
    override func viewDidLoad() {
        super.viewDidLoad()
        camera.setSampleBufferDelegate(self)
        camera.start()
        tracker.startGraph()
        tracker.delegate = self
        xyLabel.font = xyLabel.font.withSize(60)
        frameTimeLabel.font = frameTimeLabel.font.withSize(25)
        inferenceClsLabel.font = inferenceClsLabel.font.withSize(25)
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        tracker.processVideoFrame(pixelBuffer)

        DispatchQueue.main.async {
            if !self.toggleView.isOn {
                self.imageView.image = UIImage(ciImage: CIImage(cvPixelBuffer: pixelBuffer!))
            }
        }
    }
    
    
    func handTracker(_ handTracker: HandTracker!, didOutputLandmarks landmarks: [Landmark]!, andHand handSize: CGSize) {
        //Create MLMultiArray with dimensions matching the Core ML model
        let modelInput = try! MLMultiArray(shape: [21, 3], dataType: .float32)
        //Feed the landmarks into the input Array
        for i in 0...((modelInput.shape[0].intValue)-1) {
            modelInput[3*i+0] = landmarks[i].x as NSNumber
            modelInput[3*i+1] = landmarks[i].y as NSNumber
            modelInput[3*i+2] = landmarks[i].z/1000 as NSNumber
        }
        //Make a prediction and measure the time used
        self.timeClsBefore = CACurrentMediaTime();
        let modelOutput = try! model.prediction(input_1: modelInput)
        self.timeClsAfter = CACurrentMediaTime();
        //Calculate the average over 10 prediction times
        self.timeClsBuf.remove(at: 9)
        self.timeClsBuf.insert(Int(1000000*(self.timeClsAfter-self.timeClsBefore)), at: 0)
        var timeClsAvg = 0
        for i in 0...9{
            timeClsAvg = timeClsAvg + timeClsBuf[i]
        }
        timeClsAvg = timeClsAvg/10
        //Search the maximum in the prediction output and print the coresponding letter
        var maxIdx = 0
        for i in 1...((modelOutput.Identity.shape[1].intValue)-1) {
            if(modelOutput.Identity[i].floatValue > modelOutput.Identity[maxIdx].floatValue){
                maxIdx = i
            }
        }
        print("Letter: ", self.classes[maxIdx])
        //Measure the time used to process one frame
        self.timeOld = self.time1
        self.time1 = CACurrentMediaTime();
        //Calculate the average over 10 frame times
        infBuf.remove(at: 9)
        infBuf.insert(Int(1000*(self.time1-self.timeOld)), at: 0)
        var infAvg = 0
        for i in 0...9{
            infAvg = infAvg + infBuf[i]
        }
        infAvg = infAvg/10
        self.infCounter = self.infCounter + 1
        // Update the letter label every cycle and the other two labels every 10 cycles
        DispatchQueue.main.async {
            if(modelOutput.Identity[maxIdx].floatValue>8){
                self.xyLabel.text = String(self.classes[maxIdx])
            } else {
                self.xyLabel.text = "--"
            }
            if(self.infCounter == 10){
                self.frameTimeLabel.text = "Processing time of one frame: " + String(infAvg) + "ms"
                self.inferenceClsLabel.text = "Inference of classifier: " + String(timeClsAvg) + "us"
                self.infCounter = 0
            }
        }
    }
    
    func handTracker(_ handTracker: HandTracker!, didOutputPixelBuffer pixelBuffer: CVPixelBuffer!) {
        DispatchQueue.main.async {
            //Toggle button to show mediapipe visualizations
            if self.toggleView.isOn {
                self.imageView.image = UIImage(ciImage: CIImage(cvPixelBuffer: pixelBuffer))
            }
        }
    }
}

extension Collection {
    /// Returns the element at the specified index if it is within bounds, otherwise nil.
    subscript (safe index: Index) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}

extension CGFloat {
    func ceiling(toDecimal decimal: Int) -> CGFloat {
        let numberOfDigits = CGFloat(abs(pow(10.0, Double(decimal))))
        if self.sign == .minus {
            return CGFloat(Int(self * numberOfDigits)) / numberOfDigits
        } else {
            return CGFloat(ceil(self * numberOfDigits)) / numberOfDigits
        }
    }
}

extension Double {
    func ceiling(toDecimal decimal: Int) -> Double {
        let numberOfDigits = abs(pow(10.0, Double(decimal)))
        if self.sign == .minus {
            return Double(Int(self * numberOfDigits)) / numberOfDigits
        } else {
            return Double(ceil(self * numberOfDigits)) / numberOfDigits
        }
    }
}

