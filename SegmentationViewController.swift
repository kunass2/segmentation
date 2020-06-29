//
//  SegmentationViewController.swift
//  Nails
//
//  Created by Bartłomiej Semańczyk on 26/06/2020.
//  Copyright © 2020 Bartłomiej Semańczyk. All rights reserved.
//
import Vision
class SegmentationViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let mainView: SegmentationView
    private let viewModel: SegmentationViewModelable
    private let bag = DisposeBag()
    // MARK: - Initialization
    init(view: SegmentationView, viewModel: SegmentationViewModelable) {
        self.mainView = view
        self.viewModel = viewModel
        super.init(nibName: nil, bundle: nil)
    }
    required init?(coder: NSCoder) {
        return nil
    }
    // MARK: - Life Cycle
    override func viewDidLoad() {
        super.viewDidLoad()
        setupAVCapture()
        setupOverlayLayer()
        setupVisionModel()
        startCaptureSession()
    }
    // MARK: - Private
    
    private var processingSpeed: UILabel!
    
    private var segmentationOverlay: CALayer! = nil
    
    private var requests = [VNRequest]()
    
    //new variables
    var bufferSize: CGSize = .zero
    var rootLayer: CALayer! = nil
    private let session = AVCaptureSession()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let videoDataOutputQueue = DispatchQueue(label: "videoQueue", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
    private var previewLayer: AVCaptureVideoPreviewLayer! = nil
    

    func setupOverlayLayer(){
        segmentationOverlay = CALayer()
        segmentationOverlay.name = "Segmentation Overlay"
        segmentationOverlay.bounds = previewLayer.bounds
        segmentationOverlay.position = previewLayer.position
        previewLayer.addSublayer(segmentationOverlay)
    }
    
    func startCaptureSession(){
        session.startRunning()
    }
    
    func setupAVCapture(){
        var deviceInput: AVCaptureDeviceInput!
        
        //select a video device
        let videoDevice = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera],
                                                           mediaType: .video,
                                                           position: .back).devices.first
        
        do {
            deviceInput = try AVCaptureDeviceInput(device: videoDevice!)
        } catch {
            print("Could not create video device input: \(error)")
            return
        }
        
        session.beginConfiguration()
        session.sessionPreset = .vga640x480//.photo //.vga640x480
        
        //add video nput
        guard session.canAddInput(deviceInput) else{
            print("Could not add video device input to the session")
            session.commitConfiguration()
            return
        }
        
        session.addInput(deviceInput)
        if session.canAddOutput(videoDataOutput) {
            session.addOutput(videoDataOutput)
            
            //add video data output
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            //videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_24RGB)]
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        } else{
            print("Could not add video data output to the session")
            session.commitConfiguration()
            return
        }
        
        let captureConnection = videoDataOutput.connection(with: .video)
        //always process the frames
        captureConnection?.isEnabled = true
        do{
            try videoDevice!.lockForConfiguration()
            let dimensions = CMVideoFormatDescriptionGetDimensions((videoDevice?.activeFormat.formatDescription)!)
            bufferSize.width = CGFloat(dimensions.width)
            bufferSize.height = CGFloat(dimensions.height)
            videoDevice!.unlockForConfiguration()
        } catch{
            print(error)
        }
        
        session.commitConfiguration()
        
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        rootLayer = view.layer
        rootLayer.bounds = view.bounds
        previewLayer.frame = rootLayer.bounds
        rootLayer.addSublayer(previewLayer)
    }
    
    func setupVisionModel() {
        guard let visionModel = try? VNCoreMLModel(for: DeepLabV3().model) else{
            fatalError("Can not load CNN model")
        }
        
        let segmentationRequest = VNCoreMLRequest(model: visionModel, completionHandler: {request, error in
            DispatchQueue.main.async(execute: {
                if let results = request.results {
                    self.processClassification(results)
                }
            })
        })
        segmentationRequest.imageCropAndScaleOption = .scaleFill
        self.requests = [segmentationRequest]
    }
    func processClassification(_ observations: [Any]){
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        segmentationOverlay.sublayers = nil // remove previos layers
        
        //let startTime = DispatchTime.now()
        
        let obs = observations as! [VNCoreMLFeatureValueObservation] //[VNObservation]
        
        if obs.isEmpty{
            print("Empty")
        }
        //extract multi array from the observation
        let obsVal: VNCoreMLFeatureValueObservation = (obs.first)!
        let outMultiArray: MLMultiArray = obsVal.featureValue.multiArrayValue!
        
        //post process the multiarray to create mask
        let segMask = postProcess(features: outMultiArray)
        //resize the mask to match the view diemnsions
        guard let image = segMask else {
            return
        }
        let resizedMask = resizeUIImage(image: image,
                                        scaledToSize: CGSize(width: rootLayer.bounds.width, height: rootLayer.bounds.height))
        
        //add mask to CALayer and then add it as sublayer to the view
        let caLayer = CALayer()
        caLayer.frame = previewLayer.frame
        caLayer.contents = resizedMask.cgImage
        caLayer.opacity = 0.5
        
        segmentationOverlay.addSublayer(caLayer)
        CATransaction.commit()
        
        //compute processing time
        //let endTime = DispatchTime.now()
        //let timeDiff = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
        //let timeInMS = Double(timeDiff) / 1_000_000
        //self.processingSpeed.text = String(format: "Segmentation Time: %.2f ms", timeInMS)
    }
    
    func resizeUIImage(image:UIImage, scaledToSize newSize:CGSize) -> UIImage{
        UIGraphicsBeginImageContextWithOptions(newSize, false, 0.0);
        image.draw(in: CGRect(origin: CGPoint.zero, size: CGSize(width: newSize.width, height: newSize.height)))
        let newImage:UIImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return newImage
    }
    
    func pix2RGB(pix: Int) -> (UInt8, UInt8, UInt8){
        //['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        //'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        //'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        //'tv/monitor']
        switch pix {
        case 0: //backgr
            return (0, 0, 0)
        case 1: // aeroplane
            return (0, 0, 0)//(128, 0, 0)
        case 2: //bicycle
            return (64, 128, 128) //(0, 128, 0)
        case 3: //bird
            return (192, 0, 128) //(128, 128, 0)
        case 4: //boat
            return (0, 0, 0) //(0, 0, 128)
        case 5: //bottle
            return (128, 0, 128)
        case 6: //bus
            return (128, 128, 128) //(0, 128, 128)
        case 7: //car
            return (128, 128, 128)
        case 8: //cat
            return (192, 0, 128) //(64, 0, 0)
        case 9: //chair
            return (192, 0, 0) //(192, 0, 0)
        case 10: //cow
            return (192, 0, 128) //(64, 128, 0)
        case 11: //dinningtable
            return (0, 0, 0) //(192, 128, 0)
        case 12: //dog
            return (192, 0, 128) //(64, 0, 128)
        case 13: //horse
            return (192, 0, 128) //(192, 0, 128)
        case 14: //motorbike
            return (64, 128, 128)
        case 15: //person
            return (192, 128, 128)
        case 16: //potted plant
            return (0, 64, 0)
        case 17: //sheep
            return (192, 0, 128) //(128, 64, 0)
        case 18: //sofa
            return (192, 0, 0) //(0, 192, 0)
        case 19: //train
            return (128, 128, 128) //(128, 192, 0)
        case 20: //tv
            return (0, 0, 0)//(0, 64, 128)
        default:
            return (0, 0, 0)
        }
    }
    func postProcess(features: MLMultiArray) -> UIImage? {
        print(features.shape.count)
//        assert (features.shape.count == 5)
        //Seq x B x C x H x W
        let (h, w) = (features.shape[0].intValue, features.shape[1].intValue) //BEFORE 3, 4
        let bytesPerComponent = MemoryLayout<UInt8>.size
        let channels = 3 // RGB Image
        let bytesPerPixel = channels * bytesPerComponent
        let count = w * h * bytesPerPixel
        
        //var pixels = [UInt8](repeating: 0, count: count)
        var data = Data(count: count)
        
        data.withUnsafeMutableBytes { (bytes: UnsafeMutablePointer<UInt8>) -> Void in
            var pointer = bytes
            for i in 0..<w{
                for j in 0..<h{
                    let pixIndex = i * w + j
                    let pixValue = features[pixIndex].intValue
                    let (r, g, b) = pix2RGB(pix: pixValue)
                    pointer.pointee = r
                    pointer += 1
                    pointer.pointee = g
                    pointer += 1
                    pointer.pointee = b
                    pointer += 1
                }
            }
        }
        let provider : CGDataProvider = CGDataProvider(data: data as CFData)!
        let cgImg = CGImage(width: w,
                            height: h,
                            bitsPerComponent: bytesPerComponent * 8,
                            bitsPerPixel: bytesPerPixel * 8,
                            bytesPerRow: bytesPerPixel * w,
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Little.rawValue),
                            provider: provider,
                            decode: nil,
                            shouldInterpolate: false,
                            intent: CGColorRenderingIntent.defaultIntent)
        let orientation = getUIImageOrientationFromDevice()
        return UIImage(cgImage: cgImg!, scale: 1.0, orientation: orientation)
    }
    
    // this function notifies AVCatpreuDelegate everytime a new frame is received
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {return}
        let exifOrientation = getCGImageOrientationFromDevice()

        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: exifOrientation, options: [:]) //orientation: 1 orientation: orientation,
        do {
            try imageRequestHandler.perform(self.requests)
        } catch{
            print(error)
        }
    }
    
    public func getUIImageOrientationFromDevice() -> UIImage.Orientation {
        switch UIDevice.current.orientation{
        case UIDeviceOrientation.portrait:
            return UIImage.Orientation.right
        case UIDeviceOrientation.portraitUpsideDown:
            return UIImage.Orientation.left
        case UIDeviceOrientation.landscapeLeft:
            return UIImage.Orientation.up // this is the base orientation
        case UIDeviceOrientation.landscapeRight:
            return UIImage.Orientation.down
        case UIDeviceOrientation.unknown:
            return UIImage.Orientation.up
        default:
            return UIImage.Orientation.up
        }
    }
    
    public func getCGImageOrientationFromDevice() -> CGImagePropertyOrientation {
        switch UIDevice.current.orientation{
        case UIDeviceOrientation.portraitUpsideDown:
            return CGImagePropertyOrientation.left
        case UIDeviceOrientation.landscapeLeft:
            return CGImagePropertyOrientation.upMirrored
        case UIDeviceOrientation.landscapeRight:
            return CGImagePropertyOrientation.down
        case UIDeviceOrientation.portrait:
            return CGImagePropertyOrientation.up
        default:
            return CGImagePropertyOrientation.up
        }
    }
}
