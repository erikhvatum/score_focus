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

            GridLayout {
                id: gridLayout1
                columns: 2
                flow: GridLayout.LeftToRight
                anchors.fill: parent

                Label { text: "Timepoint: " }

                ComboBox {
                    id: timepointComboBox
                    Layout.fillWidth: true
                }

                Label { text: "Well: " }

                SpinBox {
                    id: wellSpinBox
                    Layout.fillWidth: true
                }
            }
        }

        GroupBox {
            id: scoringBox
            title: "Scoring"
            Layout.fillWidth: true

            GridLayout {
                columns: 3
                flow: GridLayout.LeftToRight
                anchors.fill: parent

                Label { text: "BF is focused: " }

                CheckBox {

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
