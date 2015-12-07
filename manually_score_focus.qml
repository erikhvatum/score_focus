import QtQuick 2.4
import QtQuick.Controls 1.3
import QtQuick.Layouts 1.2
import QtQuick.Extras 1.4
import QtQuick.Window 2.2

import Analysis 1.0

ExperimentManualFocusScorer {
    id: experimentManualFocusScorer
    objectName: "experimentManualFocusScorer"
    width: mainLayout.implicitWidth + 2 * margin
    height: mainLayout.implicitHeight + 2 * margin
    property int margin: 11
    property alias backgroundColor: background.color

    Rectangle {
        id: background
        anchors.fill: parent
        z: -1
    }

    ColumnLayout {
        id: mainLayout
        anchors.fill: parent
        anchors.margins: margin

        GroupBox {
            id: targetBox
            title: "Target"
            Layout.fillWidth: true
            enabled: isValid

            GridLayout {
                id: gridLayout1
                columns: 2
                flow: GridLayout.LeftToRight
                anchors.fill: parent

                Label { text: "Timepoint: " }

                ComboBox {
                    id: timepointComboBox
                    Layout.fillWidth: true
                    model: timePoints
                    textRole: "display"
                    onCurrentTextChanged: {
                        print(currentText)
                        manualFocusScore.timepoint = currentText
                    }
                }

                Label { text: "Well: " }

                ComboBox {
                    id: wellComboBox
                    Layout.fillWidth: true
                    model: hatchedWellIdxs
                    textRole: "display"
                    onCurrentTextChanged: {
                        var t = parseInt(currentText)
                        print(t)
                        manualFocusScore.wellIdx = parseInt(currentText)
                    }
                }
            }
        }

        GroupBox {
            id: scoringBox
            title: "Scoring"
            Layout.fillWidth: true
            enabled: isValid

            GridLayout {
                columns: 3
                flow: GridLayout.LeftToRight
                anchors.fill: parent

                Label {
                    enabled: manualFocusScore.hasBf
                    text: "BF is focused: "
                }

                CheckBox {
                    enabled: manualFocusScore.hasBf
                }

                Item {
                    Layout.fillWidth: true
                }

                Label {
                    enabled: manualFocusScore.focusStackLen > 0
                    text: "Best stack idx: "
                }

                SpinBox {
                    enabled: manualFocusScore.focusStackLen > 0
                    minimumValue: -1
                    maximumValue: manualFocusScore.focusStackLen - 1
                }

                Item {
                    Layout.fillWidth: true
                }
            }
        }

        Item {
            id: spacer
            Layout.fillHeight: true
        }
    }
}
