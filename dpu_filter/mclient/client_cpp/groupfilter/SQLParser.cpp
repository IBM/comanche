#include "SQLParser.h"
#include <sstream>
#include <unordered_map>
#include <cctype>
#include <algorithm>
#include <iostream>

TokenType SQLParser::getTokenType(const std::string& token) {
    static const std::unordered_map<std::string, TokenType> tokenMap = {
        {"SELECT", TokenType::SELECT},
        {"FROM", TokenType::FROM},
        {"WHERE", TokenType::WHERE},
        {"AND", TokenType::AND},
        {"OR", TokenType::OR},
        {"=", TokenType::OPERATOR},
        {">", TokenType::OPERATOR},
        {"<", TokenType::OPERATOR}
        // Add more operators as needed
    };

    auto it = tokenMap.find(token);
    if (it != tokenMap.end()) {
        return it->second;
    }

    // Check if the token is a literal (integer for simplicity)
    if (!token.empty() && std::all_of(token.begin(), token.end(), ::isdigit)) {
        return TokenType::LITERAL;
    }

    // Default to column if not a recognized keyword or operator
    return TokenType::COLUMN;
}

// Helper function to convert TokenType to string
std::string tokenTypeToString(TokenType type) {
    switch (type) {
        case TokenType::SELECT: return "SELECT";
        case TokenType::FROM: return "FROM";
        case TokenType::WHERE: return "WHERE";
        case TokenType::COLUMN: return "COLUMN";
        case TokenType::LITERAL: return "LITERAL";
        case TokenType::OPERATOR: return "OPERATOR";
        case TokenType::AND: return "AND";
        case TokenType::OR: return "OR";
        case TokenType::UNKNOWN: default: return "UNKNOWN";
    }
}


std::vector<Token> SQLParser::parse(const std::string& sql) {
    std::vector<Token> tokens;
    std::istringstream stream(sql);
    std::string token;

while (stream >> token) {
    // Handling for strings (literals with spaces or special characters)
    if (token.front() == '\'' || token.front() == '\"') {
        std::string strToken = token;
        while (strToken.back() != '\'' && strToken.back() != '\"' && stream >> token) {
            strToken += " " + token;
        }
        // Correctly declare the type here for the literal case
        TokenType type = TokenType::LITERAL;
        tokens.push_back(Token{type, strToken});
        // Adjust the print statement to use 'strToken' for the value
        //std::cout << "Parsed Token: Type = " << tokenTypeToString(type) 
        //         << ", Value = " << strToken << std::endl;
    } else {
        TokenType type = getTokenType(token);
        tokens.push_back(Token{type, token});
        // Print each token for debugging
        //std::cout << "Parsed Token: Type = " << tokenTypeToString(type) 
        //          << ", Value = " << token << std::endl;
    }
}


    return tokens;
}
