# MediapipeFaceTracking_forVRC
WebカメラでVRChatのフェイストラッキングをするためのアプリです。
以下のテンプレートを使用してセットアップしたアバターに対して動作させることを想定しています。  
https://github.com/Adjerry91/VRCFaceTracking-Templates

なお、ARKitのPerfectSync用Blendshapeを持つアバターでのみ動作確認を行っているため、SRanipalなど他の形式のアバターに対する動作は確認していません。

## 動作環境
Python3.11.7で動作確認を行っています。

## 使用している推定モデル
本アプリでは、GoogleのMediapipeが提供するFace Landmarkerモデルとして、face_landmarker_v2_with_blendshapes.taskを使用しています。

## パラメータについて
本アプリでは、顔画像から推定されたBlendshape値を感度倍し、最小値から最大値の間にマッピングした後、VRCFTのBlendshape値の範囲にクランプしています。
・顔を動かしてもアバターの表情があまり動かない場合、該当するパラメータの感度を上げてください。(0以上)  
・アバターの顔を一定以上動かしたくない場合は、最小値を大きく、最大値を小さく設定してください。(例として、瞳が動きすぎないよう、デフォルトでは最小値と最大値を0<1の範囲で設定しています。)  
・最小値をVRCFTのBlendshapeの最小値よりも小さく設定することで、下限の値が出力されやすくなります。(例として、瞼を完全に閉じやすくするために、デフォルトではEyeLidの最小値が0未満に設定されています。)

## Blendshape変換の仕様
本アプリでは、ユーザの顔に対してARKitのBlendshape値を推定したのち、その値をVRCFTのパラメータに変換してOSCメッセージを送信しています。  
内部的なBlendshape値の変換は、以下のような対応で行われています。  
※どのBlendshapeがどの値に寄与しているかを示すのみであり、マッピング方法はBlendshapeごとに異なる場合があります。

| 推定されるARKit(変換元) | VRCFT(変換後) | 範囲 | 備考 |
| - | - | - | - |
| eyeLookOutLeft<br>eyeLookInLeft | EyeLeftX | -1~1 |  |
| eyeLookOutRight<br>eyeLookInRight | EyeRightX | -1~1 |  |
| eyeLookUpLeft<br>eyeLookUpRight | EyeY | -1~1 |  |
| eyeBlinkLeft | EyeLidLeft | 0~1 |  |
| eyeBlinkRight | EyeLidRight | 0~1 |  |
| eyeSquintLeft | EyeSquintLeft | 0~1 | 機能していない可能性あり |
| eyeSquintRight | EyeSquintRight | 0~1 | 機能していない可能性あり |
| browInnerUp<br>browOuterUpLeft<br>browDownLeft | BrowExpressionLeft | -1~1 |  |
| browDownRight<br>browInnerUp<br>browOuterUpRight | BrowExpressionRight | -1~1 |  |
| noseSneerLeft<br>noseSneerRight | NoseSneer | 0~1 |  |
| cheekPuff | CheekPuffLeft | 0~1 | VRCFTでは左右同時変形のみ |
| cheekPuff | CheekPuffRight | 0~1 |  |
| jawOpen | JawOpen | 0~1 |  |
| mouthClose | MouthClosed | 0~1 |  |
| jawLeft<br>jawRight | JawX | -1~1 |  |
| jawForward | JawForward | 0~1 |  |
| mouthRollUpper | LipSuckUpper | 0~1 |  |
| mouthRollLower | LipSuckLower | 0~1 |  |
| mouthFunnel | LipFunnel | 0~1 |  |
| mouthPucker | LipPucker | 0~1 |  |
| mouthUpperUpLeft<br>mouthUpperUpRight | MouthUpperUp | 0~1 |  |
| mouthLowerDownLeft<br>mouthLowerDownRight | MouthLowerDown | 0~1 | 左右同時変形のみ |
| mouthLeft<br>mouthRight | MouthX | -1~1 |  |
| mouthSmileLeft | SmileFrownLeft | -1~1 |  |
| mouthSmileRight | SmileFrownRight | -1~1 |  |
| mouthStretchLeft | MouthStretchLeft | 0~1 |  |
| mouthStretchRight | MouthStretchRight | 0~1 |  |
|  | MouthRaiserLeft |  | 本アプリでは未使用 |
|  | MouthRaiserRight |  | 本アプリでは未使用 |
| mouthPressLeft<br>mouthPressRight | MouthPress | 0~1 | →左右同時変形のみ |
|  | TongueOut | 0-1 | 本アプリでは未使用 |


## 注意事項
・本アプリの制作にはLLMの生成したコードを部分的に使用しています。  
・本アプリを使用することで発生したいかなる問題についても、作者は一切の責任を負いません。
