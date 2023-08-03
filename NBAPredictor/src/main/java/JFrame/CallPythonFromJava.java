/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package JFrame;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
/**
 *
 * @author sebte
 */
public class CallPythonFromJava {
    public static void main(String[] args) {
        try {
            ProcessBuilder pb = new ProcessBuilder("python", "NBA_predictor.py");
            Process process = pb.start();

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                System.out.println("Python program executed successfully.");
            } else {
                System.err.println("Error executing Python program. Exit code: " + exitCode);
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        
        //Read results
        String filePath = "C:\\Users\\sebte\\OneDrive\\Documents\\Projects\\NBA_Predictor\\results.txt";
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Process each line from the file
                System.out.println(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
