import QtQuick 2.4
import QtQuick.Controls 1.4
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

    manualFocusScore.onTimepointChanged: timepointComboBox.currentIndex = timepointComboBox.find(manualFocusScore.timepoint)
    manualFocusScore.onWellIdxChanged: wellComboBox.currentIndex = wellComboBox.find(manualFocusScore.wellIdx.toString())

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
                columns: 4
                flow: GridLayout.LeftToRight
                anchors.fill: parent

                Label { text: "Timepoint: " }

                ComboBox {
                    id: timepointComboBox
                    Layout.fillWidth: true
                    model: timePoints
                    textRole: "display"
                    onCurrentTextChanged: manualFocusScore.timepoint = currentText
                }

                Button {
                    text: "-"
                    onClicked: toPrevTimepoint()
                }

                Button {
                    text: "="
                    onClicked: toNextTimepoint()
                }

                Label { text: "Well: " }

                ComboBox {
                    id: wellComboBox
                    Layout.fillWidth: true
                    model: hatchedWellIdxs
                    textRole: "display"
                    onCurrentTextChanged: manualFocusScore.wellIdx = parseInt(currentText)
                }

                Button {
                    text: "["
                    onClicked: toPrevWell()
                }

                Button {
                    text: "]"
                    onClicked: toNextWell()
                }
            }
        }

        GroupBox {
            id: scoringBox
            title: "Scoring"
            Layout.fillWidth: true
            enabled: isValid

            GridLayout {
                columns: 5
                flow: GridLayout.LeftToRight
                anchors.fill: parent

                Label {
                    enabled: manualFocusScore.hasBf
                    text: "BF is focused: "
                }

                CheckBox {
                    enabled: manualFocusScore.hasBf
                    checked: manualFocusScore.bfIsFocused
                    onCheckedChanged: manualFocusScore.bfIsFocused = checked
                }

                Item { Layout.fillWidth: true }

                Button {
                    enabled: manualFocusScore.hasBf
                    text: ";"
                    onClicked: uncheckBfFocused()
                }

                Button {
                    enabled: manualFocusScore.hasBf
                    text: "'"
                    onClicked: checkBfFocused()
                }

                Label {
                    enabled: manualFocusScore.focusStackLen > 0
                    text: "Best stack idx: "
                }

                SpinBox {
                    enabled: manualFocusScore.focusStackLen > 0
                    minimumValue: -1
                    maximumValue: manualFocusScore.focusStackLen - 1
                    value: manualFocusScore.bestFocusStackIdx
                    onValueChanged: manualFocusScore.bestFocusStackIdx = value
                }

                Item { Layout.fillWidth: true }

                Button {
                    enabled: manualFocusScore.hasBf
                    text: "."
                    onClicked: toPrevFocusStackIdx()
                }

                Button {
                    enabled: manualFocusScore.hasBf
                    text: "'"
                    onClicked: toNextFocusStackIdx()
                }

                Button {
                    text: "Refresh"
                    onClicked: manualFocusScore.refresh()
                }

                Button {
                    text: "Commit"
                    onClicked: manualFocusScore.commit()
                }

                Button {
                    text: "Commit++ \\"
                    onClicked: manualFocusScore.commitAndAdvance()
                }

                Button {
                    text: "Next unscored (no commit)"
                    onClicked: manualFocusScore.advanceToNextUnscored()
                    Layout.columnSpan: 5
                }
            }
        }

        Item { Layout.fillHeight: true }
    }
}
